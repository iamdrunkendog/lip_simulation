import os
import json

PRESETS_DIR = 'presets'
TEMPLATE_FILE = 'test4.json'

# 보존할 색상 관련 키 목록
COLOR_KEYS = {
    "LIP_COLOR_HEX",
    "LIP_COLOR_RGB",
    "COLOR_OPACITY",
    "BLENDING_MODE",
    "BASE_DESATURATION",
    "VALUE_WEIGHT",
    "DEEP_COLOR",
    "COLOR"
}

def main():
    template_path = os.path.join(PRESETS_DIR, TEMPLATE_FILE)
    if not os.path.exists(template_path):
        print(f"Error: Template file {TEMPLATE_FILE} not found.")
        return

    with open(template_path, 'r', encoding='utf-8') as f:
        template_data = json.load(f)
    
    print(f"Loaded template from {TEMPLATE_FILE}")

    count = 0
    for filename in os.listdir(PRESETS_DIR):
        # '_Glossy.json'으로 끝나는 파일만 대상
        if not filename.endswith('_Glossy.json'):
            continue
            
        target_path = os.path.join(PRESETS_DIR, filename)
        
        # 기존 Glossy 프리셋 로드 (색상 정보 백업용)
        with open(target_path, 'r', encoding='utf-8') as f:
            target_data = json.load(f)
            
        # 1. 템플릿 데이터로 초기화 (새로운 제형/물리 설정 적용)
        new_data = template_data.copy()
        
        # 2. 기존 파일의 색상 정보 복원
        for k in COLOR_KEYS:
            if k in target_data:
                new_data[k] = target_data[k]
                
        # 3. 파일 저장
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
            
        print(f"Updated {filename}")
        count += 1
        
    print(f"Total {count} glossy presets updated.")

if __name__ == "__main__":
    main()
