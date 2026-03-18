"""FlowCopy Backend - AI Marketing Content Generator (Production-Ready)"""

from __future__ import annotations

import os
import json
import base64
import uuid
import time
import logging
import httpx
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError
from dotenv import load_dotenv
from jose import jwt, JWTError
import bcrypt

import db

load_dotenv()

# ── Logging ──

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("flowcopy")

# ── App setup ──

app = FastAPI(title="FlowCopy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()
logger.info("Database initialized")

# Create uploads directory
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ── Simple in-memory rate limiter ──

class RateLimiter:
    """Simple per-IP rate limiter. Resets every window_seconds."""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        # Clean old entries
        self.requests[ip] = [t for t in self.requests[ip] if now - t < self.window]
        if len(self.requests[ip]) >= self.max_requests:
            return False
        self.requests[ip].append(now)
        return True


rate_limiter = RateLimiter(max_requests=30, window_seconds=60)

# ── Auth setup ──

SECRET_KEY = os.getenv("SECRET_KEY", "flowcopy-secret-change-me-in-production")
ALGORITHM = "HS256"
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@flowcopy.com")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


# ── AI Client (OpenRouter - free models) ──
# OpenRouter is OpenAI-SDK compatible, just different base_url + model names

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
OPENROUTER_VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.0-flash-exp:free")
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT", "60"))  # seconds

_ai_client = None


def get_ai_client() -> OpenAI:
    """Lazy-init OpenRouter client (OpenAI-compatible) with timeout protection."""
    global _ai_client
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY is not set!")
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured on server")
    if _ai_client is None:
        _ai_client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=AI_TIMEOUT,
            max_retries=2,
            default_headers={
                "HTTP-Referer": os.getenv("APP_URL", "https://flowcopy.app"),
                "X-Title": "FlowCopy",
            },
        )
        logger.info(f"AI client initialized | model={OPENROUTER_MODEL} | timeout={AI_TIMEOUT}s")
    return _ai_client


def create_token(user_id: int, email: str) -> str:
    expire = datetime.utcnow() + timedelta(days=30)
    return jwt.encode({"sub": str(user_id), "email": email, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
    except (JWTError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ── Rate limit middleware ──

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Only rate-limit API endpoints, not static files
    if request.url.path.startswith("/api/"):
        client_ip = request.client.host if request.client else "unknown"
        if not rate_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please wait and try again."},
            )
    response = await call_next(request)
    return response


# ── Auth endpoints ──

class RegisterInput(BaseModel):
    email: str
    password: str
    display_name: str = ""


class LoginInput(BaseModel):
    email: str
    password: str


def _make_user_response(user: dict, token: str) -> dict:
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "display_name": user["display_name"],
            "is_admin": bool(user["is_admin"]),
            "free_credits": user["free_credits"],
            "paid_credits": user["paid_credits"],
        },
    }


@app.post("/api/register")
async def register(data: RegisterInput):
    logger.info(f"Registration attempt: {data.email}")
    if len(data.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if db.get_user_by_email(data.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    password_hash = hash_password(data.password)
    user = db.create_user(data.email, password_hash, data.display_name)
    # First user or matching admin email becomes admin
    if user["id"] == 1 or data.email == ADMIN_EMAIL:
        db.set_admin(user["id"], True)
        user["is_admin"] = 1
    token = create_token(user["id"], user["email"])
    logger.info(f"User registered: id={user['id']} email={data.email}")
    return _make_user_response(user, token)


@app.post("/api/login")
async def login(data: LoginInput):
    logger.info(f"Login attempt: {data.email}")
    user = db.get_user_by_email(data.email)
    if not user or not verify_password(data.password, user["password_hash"]):
        logger.warning(f"Failed login attempt: {data.email}")
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token(user["id"], user["email"])
    logger.info(f"User logged in: id={user['id']} email={data.email}")
    return _make_user_response(user, token)


@app.get("/api/me")
async def get_me(user: dict = Depends(get_current_user)):
    credits = db.get_credits(user["id"])
    return {
        "id": user["id"],
        "email": user["email"],
        "display_name": user["display_name"],
        "is_admin": bool(user["is_admin"]),
        "free_credits": credits["free_credits"],
        "paid_credits": credits["paid_credits"],
    }


# ── Google OAuth ──

class GoogleLoginInput(BaseModel):
    credential: str  # Google ID token


@app.post("/api/auth/google")
async def google_login(data: GoogleLoginInput):
    logger.info("Google login attempt")
    # Verify Google token via Google's API
    async with httpx.AsyncClient(timeout=10) as client_http:
        resp = await client_http.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={data.credential}"
        )
    if resp.status_code != 200:
        logger.warning("Invalid Google token received")
        raise HTTPException(status_code=401, detail="Invalid Google token")

    info = resp.json()
    # Verify audience matches our client ID
    if GOOGLE_CLIENT_ID and info.get("aud") != GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=401, detail="Token not for this app")

    email = info.get("email")
    name = info.get("name", "")
    if not email:
        raise HTTPException(status_code=401, detail="No email in Google token")

    # Find or create user
    user = db.get_user_by_email(email)
    if not user:
        random_pw = uuid.uuid4().hex
        password_hash = hash_password(random_pw)
        user = db.create_user(email, password_hash, name)
        if user["id"] == 1 or email == ADMIN_EMAIL:
            db.set_admin(user["id"], True)
            user["is_admin"] = 1
        logger.info(f"New Google user created: {email}")
    else:
        logger.info(f"Google user logged in: {email}")

    token = create_token(user["id"], user["email"])
    return _make_user_response(user, token)


# ── Guest trial ──

@app.post("/api/auth/guest")
async def guest_login():
    guest_id = uuid.uuid4().hex[:8]
    email = f"guest_{guest_id}@flowcopy.guest"
    password_hash = hash_password(uuid.uuid4().hex)
    user = db.create_user(email, password_hash, f"Guest_{guest_id}")
    # Guests get 2 free credits instead of 5
    db.use_credit(user["id"])  # 5 -> 4
    db.use_credit(user["id"])  # 4 -> 3
    db.use_credit(user["id"])  # 3 -> 2
    token = create_token(user["id"], user["email"])
    user = db.get_user_by_id(user["id"])
    logger.info(f"Guest user created: {email} (2 credits)")
    return _make_user_response(user, token)


@app.get("/api/auth/google-client-id")
async def get_google_client_id():
    return {"client_id": GOOGLE_CLIENT_ID}


# ── Channel prompts ──

CHANNEL_PROMPTS: dict[str, str] = {
    "xiaohongshu": """你是小红书爆款文案专家。根据以下产品信息，生成一篇小红书种草笔记。
要求：
- 标题用emoji开头，带数字，制造好奇心（如"❗️用了30天才敢说的真话"）
- 正文分段，每段不超过3行
- 语气真实、口语化、有代入感，像闺蜜分享
- 结尾带3-5个相关话题标签 #xxx
- 总字数300-500字""",

    "douyin": """你是抖音短视频脚本专家。根据以下产品信息，生成一个15-60秒的带货短视频脚本。
要求：
- 【开头hook】前3秒必须抓住注意力，用反常识/痛点/悬念开场
- 【痛点共鸣】描述目标用户的真实痛苦场景
- 【产品展示】自然植入产品，突出1个核心卖点
- 【行动号召】结尾引导点击购物车/评论区
- 标注每段的画面建议和预估时长""",

    "taobao": """你是淘宝/天猫详情页文案专家。根据以下产品信息，生成商品详情页的核心文案模块。
要求：
- 【主图文案】5组主图上的短文案（每组不超过10字）
- 【卖点提炼】3个核心卖点，每个卖点一句话标题+一段50字说明
- 【场景化描述】3个使用场景的生动描写
- 【信任背书】资质/成分/技术等信任元素的文案包装
- 风格：专业、简洁、有品质感""",

    "wechat_moments": """你是朋友圈营销文案专家。根据以下产品信息，生成3条不同风格的朋友圈文案。
要求：
- 第1条：故事型（讲一个真实感的用户故事）
- 第2条：干货型（分享行业知识，自然带出产品）
- 第3条：促销型（限时优惠/福利，制造紧迫感）
- 每条100-200字，配图建议
- 不要硬广感，要像朋友在分享""",

    "google_ads": """You are a Google Ads copywriting expert. Based on the product info below, generate Google Search Ads copy.
Requirements:
- 3 Responsive Search Ad variations
- Each with 5 headlines (max 30 chars each) and 3 descriptions (max 90 chars each)
- Include power words: Free, Now, Best, Guaranteed, Limited, etc.
- Focus on benefits, not features
- Include a clear CTA""",

    "facebook_ads": """You are a Facebook/Meta Ads copywriting expert. Based on the product info below, generate Facebook ad copy.
Requirements:
- 3 ad variations: 1 short-form, 1 story-form, 1 listicle
- Each with: Primary text, Headline (40 chars max), Description
- Use emotional triggers and social proof
- Include a hook in the first line
- End with clear CTA""",
}

CHANNEL_LABELS = {
    "xiaohongshu": {"zh": "小红书笔记", "en": "Xiaohongshu Post"},
    "douyin": {"zh": "抖音短视频脚本", "en": "Douyin/TikTok Script"},
    "taobao": {"zh": "淘宝详情页文案", "en": "Taobao Product Copy"},
    "wechat_moments": {"zh": "朋友圈文案", "en": "WeChat Moments"},
    "google_ads": {"zh": "Google Ads", "en": "Google Ads"},
    "facebook_ads": {"zh": "Facebook Ads", "en": "Facebook Ads"},
}


# ── Generate endpoint (production-hardened) ──

class ProductInput(BaseModel):
    product_name: str
    product_description: str
    target_audience: str
    key_selling_points: str
    price_info: str = ""
    brand_voice: str = ""
    channels: list[str]
    image_analysis: str = ""


@app.post("/api/generate")
async def generate_content(request: Request, product: ProductInput, user: dict = Depends(get_current_user)):
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Generate request | user={user['id']} | ip={client_ip} | channels={product.channels}")

    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    # Validate input
    if not product.product_name or not product.product_name.strip():
        raise HTTPException(status_code=400, detail="Product name is required")
    if not product.product_description or not product.product_description.strip():
        raise HTTPException(status_code=400, detail="Product description is required")
    if not product.channels:
        raise HTTPException(status_code=400, detail="At least one channel is required")
    if len(product.channels) > 6:
        raise HTTPException(status_code=400, detail="Maximum 6 channels per request")

    # Check credits
    credits = db.get_credits(user["id"])
    total = credits["free_credits"] + credits["paid_credits"]
    if total <= 0 and not user["is_admin"]:
        logger.info(f"User {user['id']} out of credits")
        raise HTTPException(status_code=403, detail="NO_CREDITS")

    product_context = f"""
产品名称：{product.product_name}
产品描述：{product.product_description}
目标人群：{product.target_audience}
核心卖点：{product.key_selling_points}
价格信息：{product.price_info or '未提供'}
品牌调性：{product.brand_voice or '专业、可信赖'}
"""

    if product.image_analysis:
        product_context += f"\n产品图片AI分析结果：{product.image_analysis}\n"

    results = []
    success_count = 0
    fail_count = 0

    for channel in product.channels:
        if channel not in CHANNEL_PROMPTS:
            logger.warning(f"Unknown channel requested: {channel}")
            continue
        system_prompt = CHANNEL_PROMPTS[channel]
        start_time = time.time()
        try:
            response = get_ai_client().chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请根据以下产品信息生成内容：\n{product_context}"},
                ],
                temperature=0.8,
                max_tokens=1500,
            )

            # Safely extract content — guard against empty/missing choices
            if not response.choices or len(response.choices) == 0:
                logger.error(f"AI returned empty choices for channel={channel}")
                content = "[Error] AI returned empty response. Please try again."
                fail_count += 1
            elif not response.choices[0].message:
                logger.error(f"AI returned empty message for channel={channel}")
                content = "[Error] AI returned empty message. Please try again."
                fail_count += 1
            else:
                content = response.choices[0].message.content or ""
                if not content.strip():
                    content = "[Error] AI returned blank content. Please try again."
                    fail_count += 1
                else:
                    success_count += 1

            elapsed = round(time.time() - start_time, 2)
            logger.info(f"Channel={channel} | {elapsed}s | {len(content)} chars")

        except APITimeoutError:
            elapsed = round(time.time() - start_time, 2)
            logger.error(f"TIMEOUT for channel={channel} after {elapsed}s")
            content = "[Error] AI response timed out. Please try again."
            fail_count += 1
        except APIConnectionError as e:
            logger.error(f"CONNECTION ERROR for channel={channel}: {e}")
            content = "[Error] Could not connect to AI service. Please try later."
            fail_count += 1
        except APIStatusError as e:
            logger.error(f"API ERROR for channel={channel}: status={e.status_code} body={e.body}")
            content = f"[Error] AI service error (status {e.status_code}). Please try later."
            fail_count += 1
        except Exception as e:
            logger.error(f"UNEXPECTED ERROR for channel={channel}: {type(e).__name__}: {e}")
            content = f"[Error] Generation failed: {str(e)}"
            fail_count += 1

        results.append({
            "channel": channel,
            "channel_label_zh": CHANNEL_LABELS.get(channel, {}).get("zh", channel),
            "channel_label_en": CHANNEL_LABELS.get(channel, {}).get("en", channel),
            "content": content,
        })

    logger.info(f"Generate complete | user={user['id']} | success={success_count} fail={fail_count}")

    # Deduct credit and save history
    if not user["is_admin"]:
        db.use_credit(user["id"])
    db.save_generation(user["id"], product.product_name, product.channels, results)

    updated_credits = db.get_credits(user["id"])

    return {"results": results, "credits": updated_credits}


# ── Image analysis endpoint (production-hardened) ──

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    logger.info(f"Image upload | user={user['id']} | filename={file.filename}")

    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    content = await file.read()
    file_size_mb = round(len(content) / 1024 / 1024, 2)
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    logger.info(f"Image size: {file_size_mb}MB")

    # Save file
    ext = file.filename.split(".")[-1] if file.filename and "." in file.filename else "jpg"
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = UPLOAD_DIR / filename
    with open(filepath, "wb") as f:
        f.write(content)

    # Analyze with vision model via OpenRouter
    b64_image = base64.b64encode(content).decode("utf-8")
    mime = file.content_type or "image/jpeg"

    start_time = time.time()
    try:
        response = get_ai_client().chat.completions.create(
            model=OPENROUTER_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """分析这张产品图片，提取以下信息并用JSON格式返回：
{
  "product_name": "产品名称（如果能识别）",
  "product_description": "产品的详细描述，包括外观、材质、颜色、设计风格等",
  "key_selling_points": "从图片中能观察到的卖点，用逗号分隔",
  "brand_voice": "根据产品风格推荐的品牌调性",
  "target_audience": "推荐的目标人群"
}
只返回JSON，不要其他文字。""",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=800,
        )

        # Safely extract response
        if not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            raise ValueError("Vision model returned empty response")

        analysis_text = response.choices[0].message.content or "{}"
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Image analysis complete | {elapsed}s | {len(analysis_text)} chars")

        # Try to parse JSON from the response
        analysis_text = analysis_text.strip()
        if analysis_text.startswith("```"):
            analysis_text = analysis_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        analysis = json.loads(analysis_text)

    except json.JSONDecodeError:
        logger.warning(f"Could not parse JSON from vision response, using raw text")
        analysis = {"product_description": analysis_text}
    except APITimeoutError:
        logger.error("Image analysis timed out")
        raise HTTPException(status_code=504, detail="Image analysis timed out. Please try a smaller image.")
    except APIConnectionError:
        logger.error("Could not connect to AI for image analysis")
        raise HTTPException(status_code=502, detail="Could not connect to AI service.")
    except Exception as e:
        logger.error(f"Image analysis failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

    return {"analysis": analysis, "filename": filename}


# ── History endpoint ──

@app.get("/api/history")
async def get_history(user: dict = Depends(get_current_user)):
    generations = db.get_user_generations(user["id"])
    return {"generations": generations}


# ── Brand profiles endpoints ──

class BrandProfileInput(BaseModel):
    profile_name: str
    product_name: str = ""
    product_description: str = ""
    target_audience: str = ""
    key_selling_points: str = ""
    price_info: str = ""
    brand_voice: str = ""


@app.post("/api/brands")
async def create_brand(data: BrandProfileInput, user: dict = Depends(get_current_user)):
    profile_id = db.save_brand_profile(user["id"], data.dict())
    logger.info(f"Brand profile saved | user={user['id']} | name={data.profile_name}")
    return {"id": profile_id, "message": "Brand profile saved"}


@app.get("/api/brands")
async def list_brands(user: dict = Depends(get_current_user)):
    profiles = db.get_brand_profiles(user["id"])
    return {"profiles": profiles}


@app.delete("/api/brands/{profile_id}")
async def remove_brand(profile_id: int, user: dict = Depends(get_current_user)):
    db.delete_brand_profile(profile_id, user["id"])
    return {"message": "Deleted"}


# ── Admin endpoints ──

@app.get("/api/admin/stats")
async def admin_stats(user: dict = Depends(require_admin)):
    return db.get_admin_stats()


@app.get("/api/admin/users")
async def admin_users(user: dict = Depends(require_admin)):
    users = db.get_all_users()
    return {"users": [{
        "id": u["id"],
        "email": u["email"],
        "display_name": u["display_name"],
        "is_admin": bool(u["is_admin"]),
        "free_credits": u["free_credits"],
        "paid_credits": u["paid_credits"],
        "generation_count": u["generation_count"],
        "created_at": u["created_at"],
    } for u in users]}


class AddCreditsInput(BaseModel):
    user_id: int
    amount: int


@app.post("/api/admin/add-credits")
async def admin_add_credits(data: AddCreditsInput, user: dict = Depends(require_admin)):
    db.add_credits(data.user_id, data.amount)
    logger.info(f"Admin {user['id']} added {data.amount} credits to user {data.user_id}")
    return {"message": f"Added {data.amount} credits to user {data.user_id}"}


@app.get("/api/channels")
async def list_channels():
    return [{"id": k, "label_zh": v["zh"], "label_en": v["en"]} for k, v in CHANNEL_LABELS.items()]


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": OPENROUTER_MODEL,
        "vision_model": OPENROUTER_VISION_MODEL,
        "api_key_set": bool(os.getenv("OPENROUTER_API_KEY")),
    }


# ── Serve frontend ──

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


# Serve uploaded images
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# ── Startup log ──

@app.on_event("startup")
async def startup_log():
    port = os.environ.get("PORT", "8000")
    logger.info("=" * 50)
    logger.info("FlowCopy server starting")
    logger.info(f"  PORT: {port}")
    logger.info(f"  MODEL: {OPENROUTER_MODEL}")
    logger.info(f"  VISION: {OPENROUTER_VISION_MODEL}")
    logger.info(f"  API KEY: {'SET' if os.getenv('OPENROUTER_API_KEY') else 'MISSING!'}")
    logger.info(f"  ADMIN: {ADMIN_EMAIL}")
    logger.info(f"  TIMEOUT: {AI_TIMEOUT}s")
    logger.info("=" * 50)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
