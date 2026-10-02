#!/bin/sh
# restart-dsh.sh —— 重启 dsh 容器并在重启后自检（供 dsh-search-bridge 部署/维护使用）
#
# 用法（在 OpenWrt 宿主上）：
#   nohup /data/dsh-search-bridge/restart-dsh.sh >/dev/null 2>&1 &
#   （默认延迟 40 秒执行，给调用方留出把话说完/保存状态的时间）
#   nohup /data/dsh-search-bridge/restart-dsh.sh 10 >/dev/null 2>&1 &   # 自定义延迟秒数
#
# 自检结果追加写入 /data/dsh-search-bridge/restart-check.log

LOG=/data/dsh-search-bridge/restart-check.log
DELAY=${1:-40}

echo "[restart-dsh] $(date) 将在 ${DELAY}s 后重启 dsh 容器" >>"$LOG"
sleep "$DELAY"

echo "[restart-dsh] $(date) 执行 docker restart dsh" >>"$LOG"
docker restart dsh >>"$LOG" 2>&1
sleep 25

{
  echo "=== $(date) 重启后自检 ==="
  docker ps --filter name=dsh --format 'dsh 容器: {{.Names}} | {{.Status}}'
  curl -s -o /dev/null -w 'DSH Web(3080): HTTP %{http_code}\n' -m 15 http://127.0.0.1:3080
  curl -s -o /dev/null -w '搜索服务(8090): HTTP %{http_code}\n' -m 10 http://127.0.0.1:8090/healthz
  echo
} >>"$LOG" 2>&1

echo "[restart-dsh] $(date) 自检完成，详情见 $LOG" >>"$LOG"