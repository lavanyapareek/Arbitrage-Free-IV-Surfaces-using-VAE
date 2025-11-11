#!/usr/bin/env python3
"""Test Gemini API and list available models"""

import os
import google.generativeai as genai

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("❌ GEMINI_API_KEY not set")
    exit(1)

print("Testing Gemini API...")
print(f"API Key: {api_key[:20]}...")

genai.configure(api_key=api_key)

print("\n📋 Available models:")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"  ✅ {m.name}")
except Exception as e:
    print(f"❌ Error: {e}")
    
print("\n🧪 Testing a simple generation...")
try:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Say hello!")
    print(f"✅ Success! Response: {response.text}")
except Exception as e:
    print(f"❌ Error: {e}")
