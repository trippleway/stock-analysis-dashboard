#!/bin/bash
# 每日美股分析 + Dashboard 一鍵啟動腳本

cd "$(dirname "$0")"

echo "=========================================="
echo "美股分析 Dashboard 啟動"
echo "=========================================="

# 第一步：跑 daily_picker 產生當日分析（含快取）
echo ""
echo "[1/2] 執行每日選股分析（第一次約 5-8 分鐘，後續快取秒過）..."
python src/daily_picker.py

if [ $? -ne 0 ]; then
    echo "❌ daily_picker 執行失敗，請檢查錯誤訊息"
    exit 1
fi

# 第二步：啟動 Dashboard
echo ""
echo "[2/2] 啟動 Dashboard..."
echo "=========================================="
echo "Dashboard 網址：http://localhost:8501"
echo "按 Ctrl+C 可停止"
echo "=========================================="
streamlit run dashboard/app.py
