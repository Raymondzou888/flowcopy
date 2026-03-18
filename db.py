"""FlowCopy Database - SQLite operations"""

from __future__ import annotations

import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "flowcopy.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT DEFAULT '',
            language TEXT DEFAULT 'zh',
            is_admin INTEGER DEFAULT 0,
            free_credits INTEGER DEFAULT 5,
            paid_credits INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            product_name TEXT,
            channels TEXT,
            results TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS brand_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            profile_name TEXT NOT NULL,
            product_name TEXT DEFAULT '',
            product_description TEXT DEFAULT '',
            target_audience TEXT DEFAULT '',
            key_selling_points TEXT DEFAULT '',
            price_info TEXT DEFAULT '',
            brand_voice TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()


# ── User operations ──

def create_user(email: str, password_hash: str, display_name: str = "") -> dict:
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (email, password_hash, display_name) VALUES (?, ?, ?)",
            (email, password_hash, display_name),
        )
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(user)
    finally:
        conn.close()


def get_user_by_email(email: str) -> dict | None:
    conn = get_db()
    try:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return dict(user) if user else None
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict | None:
    conn = get_db()
    try:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(user) if user else None
    finally:
        conn.close()


def use_credit(user_id: int) -> bool:
    conn = get_db()
    try:
        user = conn.execute("SELECT free_credits, paid_credits FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            return False
        if user["free_credits"] > 0:
            conn.execute("UPDATE users SET free_credits = free_credits - 1 WHERE id = ?", (user_id,))
            conn.commit()
            return True
        elif user["paid_credits"] > 0:
            conn.execute("UPDATE users SET paid_credits = paid_credits - 1 WHERE id = ?", (user_id,))
            conn.commit()
            return True
        return False
    finally:
        conn.close()


def get_credits(user_id: int) -> dict:
    conn = get_db()
    try:
        user = conn.execute("SELECT free_credits, paid_credits FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            return {"free_credits": 0, "paid_credits": 0}
        return {"free_credits": user["free_credits"], "paid_credits": user["paid_credits"]}
    finally:
        conn.close()


def add_credits(user_id: int, amount: int) -> bool:
    conn = get_db()
    try:
        conn.execute("UPDATE users SET paid_credits = paid_credits + ? WHERE id = ?", (amount, user_id))
        conn.commit()
        return True
    finally:
        conn.close()


def set_admin(user_id: int, is_admin: bool = True):
    conn = get_db()
    try:
        conn.execute("UPDATE users SET is_admin = ? WHERE id = ?", (1 if is_admin else 0, user_id))
        conn.commit()
    finally:
        conn.close()


# ── Generation history ──

def save_generation(user_id: int, product_name: str, channels: list, results: list):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO generations (user_id, product_name, channels, results) VALUES (?, ?, ?, ?)",
            (user_id, product_name, json.dumps(channels), json.dumps(results, ensure_ascii=False)),
        )
        conn.commit()
    finally:
        conn.close()


def get_user_generations(user_id: int, limit: int = 20) -> list:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM generations WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["channels"] = json.loads(d["channels"]) if d["channels"] else []
            d["results"] = json.loads(d["results"]) if d["results"] else []
            result.append(d)
        return result
    finally:
        conn.close()


# ── Brand profiles ──

def save_brand_profile(user_id: int, data: dict) -> int:
    conn = get_db()
    try:
        cursor = conn.execute(
            """INSERT INTO brand_profiles
               (user_id, profile_name, product_name, product_description, target_audience, key_selling_points, price_info, brand_voice)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, data.get("profile_name", ""), data.get("product_name", ""),
             data.get("product_description", ""), data.get("target_audience", ""),
             data.get("key_selling_points", ""), data.get("price_info", ""),
             data.get("brand_voice", "")),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_brand_profiles(user_id: int) -> list:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM brand_profiles WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_brand_profile(profile_id: int, user_id: int) -> bool:
    conn = get_db()
    try:
        conn.execute("DELETE FROM brand_profiles WHERE id = ? AND user_id = ?", (profile_id, user_id))
        conn.commit()
        return True
    finally:
        conn.close()


# ── Admin ──

def get_all_users() -> list:
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT u.*, COUNT(g.id) as generation_count
               FROM users u LEFT JOIN generations g ON u.id = g.user_id
               GROUP BY u.id ORDER BY u.created_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_admin_stats() -> dict:
    conn = get_db()
    try:
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        today = datetime.utcnow().strftime("%Y-%m-%d")
        today_users = conn.execute(
            "SELECT COUNT(*) FROM users WHERE created_at LIKE ?", (f"{today}%",)
        ).fetchone()[0]
        total_generations = conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
        today_generations = conn.execute(
            "SELECT COUNT(*) FROM generations WHERE created_at LIKE ?", (f"{today}%",)
        ).fetchone()[0]
        paid_users = conn.execute(
            "SELECT COUNT(*) FROM users WHERE paid_credits > 0 OR free_credits < 5"
        ).fetchone()[0]
        return {
            "total_users": total_users,
            "today_users": today_users,
            "total_generations": total_generations,
            "today_generations": today_generations,
            "active_users": paid_users,
        }
    finally:
        conn.close()
