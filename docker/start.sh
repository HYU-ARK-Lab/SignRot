#!/usr/bin/env bash
# RunPod pod entrypoint: expose container env to ssh sessions, start sshd, idle.
set -e

# ssh sessions don't inherit docker ENV — persist everything relevant
{
    printenv | while IFS='=' read -r key value; do
        case "$key" in
            HOME|PWD|SHLVL|TERM|_) continue ;;
        esac
        printf 'export %s=%q\n' "$key" "$value"
    done
} > /etc/rp_environment
grep -qxF 'source /etc/rp_environment' /root/.bashrc || echo 'source /etc/rp_environment' >> /root/.bashrc

mkdir -p "${HF_HOME:-/workspace/.cache/huggingface}"

# RunPod injects PUBLIC_KEY when ssh access is enabled on the pod
if [[ -n "$PUBLIC_KEY" ]]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

mkdir -p /run/sshd
/usr/sbin/sshd

echo "Pod ready. Project at /app, venv already on PATH."
sleep infinity
