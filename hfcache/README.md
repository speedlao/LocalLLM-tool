# hfcache — 自建 HuggingFace Hub 模型下载工具

单文件 stdlib 实现（Python ≥ 3.10，零第三方依赖）。
按 **repo_id + commit** 精确管理下载任务，落盘严格兼容官方
`~/.cache/huggingface/hub` 的 `refs / blobs / snapshots / trees` 约定，
在此之上加了一层内容寻址存储 `cas/` 与 SQLite 任务状态机。

## 快速开始

```powershell
# 0) 体检：确认缓存位置、代理、链接能力
python hfcache.py doctor

# 1) 看清单：某 commit 该有哪些文件、各多大、状态如何
python hfcache.py plan openai-community/gpt2 --inc "*.json" --inc "*.safetensors"

# 2) 下载
python hfcache.py sync facebook/opt-125m --rev main --jobs 4

# 3) 事后检查
python hfcache.py status facebook/opt-125m          # 离线优先
python hfcache.py verify facebook/opt-125m          # 逐文件重算哈希
python hfcache.py prune --dry-run                   # 清理残留分片 / 无引用对象

# 指定 commit 而不是分支
python hfcache.py sync openai-community/gpt2 --rev 607a30d783dfa663caf39e06633721c8d4cfcd7e
```

常用参数：`--cache-dir` `--endpoint` `--proxy http://host:port` `--token` `--jobs` `--max-retry` `--breaker` `--quiet`
环境变量：`HF_HUB_CACHE` `HF_ENDPOINT` `HTTPS_PROXY` `HF_TOKEN` `HFCACHE_JOBS` `HFCACHE_MAX_RETRY` `HFCACHE_BREAKER`

## 落盘结构

```
<CACHE>/
├── CACHEDIR.TAG                       # 告诉备份工具跳过
├── cas/<xx>/<content_sha256>          # ★ 字节唯一来源，跨仓库/跨 commit 去重
├── .tool/journal.sqlite3              # 任务状态机
└── models--<ns>--<name>/
    ├── refs/main                      # 裸 40 位 commit hash（离线解析用）
    ├── blobs/<etag>                   # 硬链接 → cas 对象（命名完全按官方 etag）
    ├── snapshots/<sha>/<path>         # 硬链接 → blobs，结构与官方一致
    └── trees/<sha>.json               # commit 文件清单，永久缓存
```

硬链接验证（同一 inode）：`cas == blobs == snapshots`，`st_nlink == 3`。
被剪掉的中间 blob 在 prune 里靠 `st_nlink == 1` 判定"无引用"。

## 五条与实现相关的关键事实（实测得出，不是文档摘抄）

1. **元数据只出现在 302 那一跳。**
   `HEAD /{repo}/resolve/{sha}/{path}` 返回 302 时带 `X-Repo-Commit`、`X-Linked-Etag`、`X-Linked-Size`；
   若像普通 HTTP 客户端那样自动跟到 CDN，最终响应里 `X-Repo-Commit` 会**消失**，
   且 `ETag` 会退化成 `xetHash`——它**不是** blob 该用的 etag。
   所以 HEAD 取元数据时锁定"第一个带元数据的跳点"，GET 才允许跟完整跳转链。
   工具里的 `Http.head()` 就是按这个纪律写的。

2. **blob 文件名的哈希语义分两种，别自己算错。**
   - 非 LFS 文件：etag = git blob oid = `sha1(b"blob <size>\0" + 内容)`，是 **40 位**。
     （例：`config.json` 的 etag `10c66461…`，等于内容 `sha256` `0daed774…` 是另一回事。）
   - LFS 文件：etag = 内容的 **sha256**，是 64 位。
   两种都直接拿服务端返回值当文件名，工具只做校验不做推导。

3. **`trees/<commit>.json` 是缓存里最值钱的东西。**
   commit 不可变 → 清单永不失效。有它就意味着：列任务、判断快照完整性
   **一个 HTTP 请求都不用发**，这也是 `status` 能做到离线跑的原因。
   格式沿用官方 `format_version: 1`。

4. **commit hash 只在所属仓库内有意义。**
   任务主键必须是 `(repo_id, revision, path)` 三元组，工具里的 `tasks.id` 就是这么算的。

5. **本机不能用符号链接，硬链接可以。**
   官方共享 blob 存储建立在 symlink 上，而这台机器 `os.symlink` 直接报
   `WinError 1314`（开发者模式未开）。所以本工具改用 hardlink：
   读方完全透明，`open()`/`os.path.exists` 都当普通文件看，但 prune 必须
   按 `st_nlink` 而非"链接是否悬空"来判定引用。跨卷时自动退化为复制。

## 与官方缓存的互操作

`blobs/snapshots/trees/refs` 目录结构与语义保持一致，官方工具链
（`huggingface_hub`、`transformers`、`vllm` 等）可以直接读；
差异只在 `cas/` 这层是额外目录，官方工具会把它当普通子目录忽略，
`hf cache ls` 有可能报 `unknown entries`——不影响模型加载。

反过来，本工具读取的 etag 与哈希规则与官方一致，因此从别处
下载进来的 blob 也能被正确识别为"已缓存"。

## 已知边界

- 仅支持 `models` 仓库类型（datasets/spaces 未覆盖）。
- `prune` 只清 `.part` 与 `st_nlink==1` 的 cas 对象，不会替你删 snapshot 目录。
- 私有 / gated 仓库需要 `--token`，工具不读 `~/.cache/huggingface/token`，
  建议显式传 `--token` 或 `HF_TOKEN`。
- 下载并发上限是本机网络瓶颈，默认 8；单文件重试 5 次，
  连续失败 5 次触发熔断（`--breaker`）。
