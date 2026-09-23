# PyPI 发布步骤（Trusted Publishing，不需要 token）

本仓库已配好 `.github/workflows/release.yml`：**打 tag 即自动发布**。
唯一需要人工做的是在 PyPI 网页上登记一次。

`alphalens-cna` 这个名字已确认未被占用（PyPI 返回 404）。

---

## 一、你需要做的（约 5 分钟，只需一次）

1. **注册账号** → <https://pypi.org/account/register/>
   邮箱验证即可，免费。

2. **开启两步验证（2FA）** —— PyPI 强制要求，否则无法发布。
   用手机上的验证器 App（Google Authenticator / 1Password / 微软验证器都行）扫码。
   在 <https://pypi.org/manage/account/totp-provisioning/>

3. **登记 Trusted Publisher**（**不需要生成任何 token**）
   打开 <https://pypi.org/manage/account/publishing/>，
   在 **"Add a pending publisher"**（项目还没建，所以是 pending）里填：

   | 字段 | 填什么 |
   |---|---|
   | PyPI Project Name | `alphalens-cna` |
   | Owner | `yinxiuqu` |
   | Repository name | `alphalens-cna` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   填完保存即生效 —— 之后 GitHub Actions 就能用 OIDC 直接发布，
   **密钥不经过任何人的手**（我不需要、也拿不到你的 token）。

## 二、然后发布（每次发版都这样）

```bash
# 1. 改版本号（pyproject.toml 的 version）
# 2. 更新 CHANGELOG.md
git commit -am "release: v0.1.0"
git tag v0.1.0
git push origin main --tags
```

推送 tag 后 GitHub Actions 会自动：构建 sdist + wheel → 发布到 PyPI。
去 <https://pypi.org/project/alphalens-cna/> 看到即成功。

## 三、发布前自查（本地）

```bash
pip install build twine
python -m build                     # 产出 dist/*.whl 与 *.tar.gz
twine check dist/*                  # 检查 metadata 是否合规
pip install dist/*.whl --target /tmp/t && \
  PYTHONPATH=/tmp/t python -c "import alphalens_cna; print(len(alphalens_cna.__all__))"
```

## 四、先练手（可选）

想先验证流程，可发到 **TestPyPI**（独立账号，<https://test.pypi.org/>）：
在那边同样登记一次 Trusted Publisher，把 `release.yml` 的 `repository-url`
指向 `https://test.pypi.org/legacy/` 即可。

## 五、常见坑

- **版本号已存在**：PyPI 不允许覆盖，必须 bump 版本
- **README 渲染失败**：`pyproject.toml` 的 `readme` 指向的文件必须存在且编码 UTF-8
- **workflow 名字不匹配**：Trusted Publisher 里填的 workflow 文件名必须与
  `.github/workflows/release.yml` **完全一致**（含后缀）
- **environment 不匹配**：`release.yml` 里写的 `environment: pypi` 要与网页一致
