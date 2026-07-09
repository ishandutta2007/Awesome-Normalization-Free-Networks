import os

with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

badges = '<a href="https://github.com/ishandutta2007/Awesome-Awesome-Awesome"><img src="https://img.shields.io/badge/Awesome-%E2%9C%94-blueviolet?style=flat-square&logo=github" alt="Awesome"/></a> <a href="https://discord.gg/jc4xtF58Ve"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord" /></a>'

# Find the spot just after the banner
target = "![Banner](assets/banner.svg)\n"
replacement = target + f"\n<div align=\"center\">\n  {badges}\n</div>\n\n"

# Add SEO text
seo_text = "<!-- SEO Meta Tags:\nKeywords: Normalization-Free Networks, NFNets, Adaptive Gradient Clipping, Deep Learning, ResNet, Vision Transformers, PyTorch, Computer Vision\nDescription: A curated awesome list and technical guide on Normalization-Free Networks (NFNets), exploring Adaptive Gradient Clipping, architecture variants like NF-ResNets, and real-world AI applications in deep learning without batch normalization.\n-->\n"
content = seo_text + content.replace(target, replacement)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(content)
