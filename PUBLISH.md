# 发布流程

## 前置准备
- 确认已配置私有注册表 `ktra`（用户级配置路径：`%USERPROFILE%\.cargo\config.toml`）。例如：

```toml
[registries.ktra]
index = "https://<你的-index-地址>"
```

- 如需鉴权，先登录：

```sh
cargo login --registry ktra <TOKEN>
```

- 更新版本号：编辑 `Cargo.toml` 中的 `version`，遵循语义化版本（如 `0.1.1`）。

## 质量检查（可选但推荐）
- 代码与依赖检查：

```sh
cargo check
```

- 运行测试（若项目包含测试）：

```sh
cargo test
```

- 打包预览，检查打包内容：

```sh
cargo package
```

## 试发布（不真正发布）

```sh
cargo publish --registry ktra --dry-run
```

## 正式发布

```sh
cargo publish --registry ktra
```

## 发布后操作（推荐）
- 为该版本创建并推送 Git 标签：

```sh
git tag v<version>
git push --tags
```

- 如需控制发布文件范围，可在 `Cargo.toml` 配置 `include`/`exclude`。

