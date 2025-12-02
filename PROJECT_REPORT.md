# 项目报告（一页）

## 项目概览
- 项目目标：基于多智能体协作生成英文短篇“图文小说”，并批量产出不同题材的成品（含插图与PDF）。
- 运行环境：Conda 环境“chemist”，Windows + PowerShell；核心依赖见 `comic_agent/requirements.txt`。
- 产出位置：`output_comics/` 下的五个子目录，每个目录包含插图（`illustration_*.png`）与组合好的 PDF 小说。

## 模型与算法
- 模型：`ogkalu/Comic-Diffusion`（SD 1.5 系）通过 `diffusers.StableDiffusionPipeline` 推理；关闭 CSA 自定义 UNet，避免形状不匹配。
- 文本代理链：
  - `ScriptWriterAgent`：生成分段叙事与插图标记（英文、短句、连贯故事）。
  - `RefinerAgent`：使用 `CharacterMemory`、`SceneMemory` 与 `PromptNormalizer`，将插图段落压缩为 10–12 词英文关键词，并加强身份词与一致性。
  - `ImageGeneratorAgent`：质量模式默认为 `high`（步数与 CFG 更高），顺序生成插图；统一负向提示词，减少面部畸变和文字水印。
  - `ComposerAgent`：通过 `fpdf2` 注册 Unicode TTF 字体，组合文本与插图为 PDF，修复 Helvetica Unicode 与加粗样式问题。
- 关键策略：
  - 全英文文本与 prompts；短促关键信息；统一强负向 prompt（避免模糊、手部异常、Logo/水印等）。
  - 种子与身份词（identity tokens）加权，提升角色一致性与场景逻辑。
  - 顺序生成（非批并行），稳定推理；目录输出按小说独立分文件夹。

## 批量结果摘要
- `output_comics/cozy_cafe`：PDF `cozy_cafe_novel.pdf` + 5 张插图；主题为夜间暖调咖啡馆的对话与启示。
- `output_comics/cyberpunk_city`：PDF `cyberpunk_city_novel.pdf` + 5 张插图；主题为雨夜街区、黑客潜入与霓虹都市。
- `output_comics/detective_blackwood`：PDF `detective_blackwood_novel.pdf` + 6 张插图；主题为维多利亚侦探与贵族庄园之谜。（已修复历史嵌套目录问题）
- `output_comics/epic_fantasy`：PDF `epic_fantasy_novel.pdf` + 6 张插图；主题为龙与王座碎片的史诗冒险。
- `output_comics/space_odyssey`：PDF `space_odyssey_novel.pdf` + 6 张插图；主题为近比邻星的外星遗迹与航行奇观。

## 质量与稳定性
- 默认质量模式：`high`；验证单次与批量运行均稳定完成。
- 已知非阻塞告警：Hugging Face 缓存中缺失 `.safetensors` 文件时，`diffusers` 回退到不安全序列化（可设 `allow_pickle=False` 强制报错）。目前生成效果正常。
- PDF 编码：通过 TTF 字体注册确保 Unicode 文本渲染与样式一致。

## 路径与使用
- 单本运行示例：
  - 入口脚本：`comic_agent/run_holmes_comic.py`（示例流）
  - 批量脚本：`comic_agent/generate_batch_novels.py`（五本不同题材）
- 运行命令（PowerShell）：
```
python .\comic_agent\generate_batch_novels.py
```

## 后续工作（建议）
- 进一步调优负向提示词与人物身份词权重，提升细节一致性。
- 对插图生成增加异常段落跳过与回退策略，增强鲁棒性。
- 如需更高画质，可在 `project_config.json` 中扩展 `quality_modes` 参数或切换更强底模。
