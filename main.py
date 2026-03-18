"""FlowCopy Backend - AI Marketing Content Generator (Full Version)"""

from __future__ import annotations

import os
import json
import base64
import uuid
import httpx
from pathlib import Path
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from jose import jwt, JWTError
import bcrypt

import db

load_dotenv()

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

# Create uploads directory
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

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

_ai_client = None

def get_ai_client() -> OpenAI:
    """Lazy-init OpenRouter client (OpenAI-compatible)."""
    global _ai_client
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured on server")
    if _ai_client is None:
        _ai_client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            default_headers={
                "HTTP-Referer": os.getenv("APP_URL", "https://flowcopy.app"),
                "X-Title": "FlowCopy",
            },
        )
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


# ── Auth endpoints ──

class RegisterInput(BaseModel):
    email: str
    password: str
    display_name: str = ""


class LoginInput(BaseModel):
    email: str
    password: str


@app.post("/api/register")
async def register(data: RegisterInput):
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


@app.post("/api/login")
async def login(data: LoginInput):
    user = db.get_user_by_email(data.email)
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token(user["id"], user["email"])
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


@app.post("/api/auth/google")
async def google_login(data: GoogleLoginInput):
    # Verify Google token via Google's API
    async with httpx.AsyncClient() as client_http:
        resp = await client_http.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={data.credential}"
        )
    if resp.status_code != 200:
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
        # Create user with a random password (they'll use Google to login)
        random_pw = uuid.uuid4().hex
        password_hash = hash_password(random_pw)
        user = db.create_user(email, password_hash, name)
        if user["id"] == 1 or email == ADMIN_EMAIL:
            db.set_admin(user["id"], True)
            user["is_admin"] = 1

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


# ── Generate endpoint ──

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
async def generate_content(product: ProductInput, user: dict = Depends(get_current_user)):
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    # Check credits
    credits = db.get_credits(user["id"])
    total = credits["free_credits"] + credits["paid_credits"]
    if total <= 0 and not user["is_admin"]:
        raise HTTPException(status_code=403, detail="NO_CREDITS")

    if not product.product_name or not product.product_description:
        raise HTTPException(status_code=400, detail="Product name and description are required")

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
    for channel in product.channels:
        if channel not in CHANNEL_PROMPTS:
            continue
        system_prompt = CHANNEL_PROMPTS[channel]
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
            content = response.choices[0].message.content or ""
        except Exception as e:
            content = f"Generation failed: {str(e)}"
        results.append({
            "channel": channel,
            "channel_label_zh": CHANNEL_LABELS.get(channel, {}).get("zh", channel),
            "channel_label_en": CHANNEL_LABELS.get(channel, {}).get("en", channel),
            "content": content,
        })

    # Deduct credit and save history
    if not user["is_admin"]:
        db.use_credit(user["id"])
    db.save_generation(user["id"], product.product_name, product.channels, results)

    updated_credits = db.get_credits(user["id"])

    return {"results": results, "credits": updated_credits}


# ── Image analysis endpoint ──

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    # Save file
    ext = file.filename.split(".")[-1] if file.filename and "." in file.filename else "jpg"
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = UPLOAD_DIR / filename
    with open(filepath, "wb") as f:
        f.write(content)

    # Analyze with vision model via OpenRouter
    b64_image = base64.b64encode(content).decode("utf-8")
    mime = file.content_type or "image/jpeg"

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
        analysis_text = response.choices[0].message.content or "{}"
        # Try to parse JSON from the response
        analysis_text = analysis_text.strip()
        if analysis_text.startswith("```"):
            analysis_text = analysis_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        analysis = json.loads(analysis_text)
    except json.JSONDecodeError:
        analysis = {"product_description": analysis_text}
    except Exception as e:
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
    return {"message": f"Added {data.amount} credits to user {data.user_id}"}


@app.get("/api/channels")
async def list_channels():
    return [{"id": k, "label_zh": v["zh"], "label_en": v["en"]} for k, v in CHANNEL_LABELS.items()]


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ── Serve frontend ──

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


# Serve uploaded images
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
