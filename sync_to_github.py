import os
import requests
import base64
import subprocess
import json
import re
import urllib3
from datetime import datetime
from pathlib import Path

# 禁用安全请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 配置区 ---
PROXY = "http://10.200.86.85:8080"
VERIFY_SSL = False 

# 从环境变量获取 Token
TOKEN = os.getenv("GITHUB_TOKEN")

def run_git_command(args):
    """封装 git 命令调用，确保路径不转义且使用 UTF-8"""
    cmd = ["git", "-c", "core.quotepath=false"] + args
    try:
        return subprocess.check_output(cmd, text=True, encoding='utf-8').strip()
    except:
        return subprocess.check_output(cmd, text=True, encoding='gbk').strip()

def get_github_remote_info():
    """自动寻找指向 github.com 的 Remote 配置"""
    try:
        output = run_git_command(["remote", "-v"])
        for line in output.splitlines():
            parts = line.split()
            if len(parts) < 2: continue
            remote_name, url = parts[0], parts[1]
            match = re.search(r"github\.com[:/]([^/]+)/([^/.]+?)(?:\.git)?(?:$|/|\s)", url)
            if match:
                return remote_name, match.group(1), match.group(2)
    except Exception as e:
        print(f"探测 GitHub Remote 失败: {e}")
    return None, None, None

def get_remote_branch_name(session, remote_name, owner, repo):
    """获取远端默认分支名"""
    try:
        head = run_git_command(["symbolic-ref", f"refs/remotes/{remote_name}/HEAD"])
        return head.split("/")[-1]
    except:
        pass
    try:
        resp = session.get(f"https://api.github.com/repos/{owner}/{repo}")
        return resp.json().get("default_branch", "master")
    except:
        return "master"

def get_git_changes(remote_name, remote_branch):
    """
    核心比对逻辑：直接对比【远程分支】与【本地工作区现状】。
    这是最稳妥的 WYSIWYG（所见即所得）模式。
    """
    change_map = {}
    try:
        # 1. 对比远程分支与当前工作区 (包括已 commit 和未 commit 的改动)
        # 结果中：A=新增, M=修改, D=删除
        output = run_git_command(["diff", f"{remote_name}/{remote_branch}", "--name-status"])
        if output:
            for line in output.splitlines():
                parts = line.split(None, 1)
                if len(parts) < 2: continue
                status, path = parts[0], parts[1].strip()
                
                if status.startswith('D'):
                    change_map[path] = "Deleted"
                elif status.startswith('A'):
                    change_map[path] = "Added (Tracked)"
                else:
                    change_map[path] = "Modified"

        # 2. 识别未跟踪的新文件 (Untracked)
        untracked = run_git_command(["ls-files", "--others", "--exclude-standard"])
        if untracked:
            for path in untracked.splitlines():
                p = path.strip()
                change_map[p] = "Added (Untracked)"
                
        return change_map
    except Exception as e:
        print(f"获取 Git 差异失败: {e}")
        return {}

def upload_blob(session, owner, repo, file_path):
    """通过 API 上传文件"""
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        b64_content = base64.b64encode(content).decode("utf-8")
        resp = session.post(f"https://api.github.com/repos/{owner}/{repo}/git/blobs", 
                           json={"content": b64_content, "encoding": "base64"})
        resp.raise_for_status()
        return resp.json()["sha"]
    except Exception as e:
        print(f"上传 {file_path} 失败: {e}")
        return None

def main():
    if not TOKEN:
        print("❌ 错误: 未检测到环境变量 GITHUB_TOKEN")
        return

    session = requests.Session()
    if PROXY: session.proxies = {"http": PROXY, "https": PROXY}
    session.verify = VERIFY_SSL
    session.headers.update({"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3+json"})

    remote_name, owner, repo = get_github_remote_info()
    if not remote_name:
        print("❌ 错误: 未能在本地配置中识别到 GitHub 仓库。")
        return

    target_branch = get_remote_branch_name(session, remote_name, owner, repo)
    curr_branch = run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])

    print(f"✅ 已匹配 GitHub: '{remote_name}' -> {owner}/{repo}")
    print(f"🔍 比对基准: 远端({remote_name}/{target_branch})")
    print(f"📤 同步内容: 本地工作区(当前分支:{curr_branch}) 的最终状态")
    
    changes = get_git_changes(remote_name, target_branch)
    if not changes:
        print("✅ 恭喜！本地工作区与 GitHub 远端已完全同步，无须操作。")
        return

    print("\n--- 待同步到远程的改动列表 ---")
    status_order = {"Deleted": 0, "Added (Untracked)": 1, "Added (Tracked)": 1, "Modified": 2}
    for path, status in sorted(changes.items(), key=lambda x: (status_order.get(x[1], 9), x[0])):
        print(f"[{status}] {path}")

    # --- 提交日志 ---
    default_msg = f"Sync from local - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    print(f"\n💬 请输入提交日志 (回车使用默认):")
    user_msg = input("> ").strip()
    commit_msg = user_msg if user_msg else default_msg

    confirm = input(f"\n❓ 确认将上述 {len(changes)} 项改动同步到 GitHub? (y/n): ").lower()
    if confirm != 'y': return

    try:
        # 获取基准
        ref_data = session.get(f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{target_branch}").json()
        last_commit_sha = ref_data["object"]["sha"]
        base_tree_sha = session.get(f"https://api.github.com/repos/{owner}/{repo}/git/commits/{last_commit_sha}").json()["tree"]["sha"]

        # 构造 Tree Items
        tree_items = []
        for f_path, status in changes.items():
            if status == "Deleted":
                # 告诉 GitHub 移除该文件
                tree_items.append({"path": f_path.replace("\\", "/"), "mode": "100644", "type": "blob", "sha": None})
                print(f"🗑️ 标记删除: {f_path}")
            else:
                if not os.path.isfile(f_path): 
                    print(f"⚠️ 跳过缺失文件: {f_path}")
                    continue
                print(f"正在上传: {f_path} ...")
                sha = upload_blob(session, owner, repo, f_path)
                if sha:
                    tree_items.append({"path": f_path.replace("\\", "/"), "mode": "100644", "type": "blob", "sha": sha})
        
        if not tree_items:
            print("❌ 无有效物理改动（可能文件已被删除但未同步），同步取消。")
            return

        # 创建远程 Commit
        print("\n✅ 处理完成。正在推送至 GitHub 接口...")
        new_tree_sha = session.post(f"https://api.github.com/repos/{owner}/{repo}/git/trees", 
                                   json={"base_tree": base_tree_sha, "tree": tree_items}).json()["sha"]
        
        new_commit_sha = session.post(f"https://api.github.com/repos/{owner}/{repo}/git/commits", 
                                     json={"message": commit_msg, "tree": new_tree_sha, "parents": [last_commit_sha]}).json()["sha"]

        resp = session.patch(f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{target_branch}", json={"sha": new_commit_sha})
        
        if resp.status_code == 200:
            print(f"\n✨ 远程同步成功！")
            print(f"🆔 Commit SHA: {new_commit_sha[:7]}")
            print(f"🔗 URL: https://github.com/{owner}/{repo}/commit/{new_commit_sha}")
        else:
            print(f"\n❌ 远程更新失败: {resp.text}")

    except Exception as e:
        print(f"\n💥 错误: {e}")

if __name__ == "__main__":
    main()
