# Starter Code — Lab #4

> Dành cho học viên VinUni AI Product Labs.

## Cách Sử Dụng

1. **Cài đặt môi trường:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Chạy bài lab:**
   - `tools.py` cung cấp tra cứu catalog, tạo support ticket và JSON Schemas.
   - `template.py` cung cấp System Prompt, baseline mock và ToolCallingAgent loop.

3. **Chạy thử:**
   ```bash
   python3 template.py
   ```

4. **Kiểm thử autograder:**
   ```bash
   python3 -m pytest ../autograder/test_agent.py -v
   ```

## Lưu Ý
- Code chạy ở chế độ **Mock Simulator** mặc định (không cần API key).
- Nếu muốn thử Live API, đặt biến môi trường: `export GEMINI_API_KEY=your_key`
