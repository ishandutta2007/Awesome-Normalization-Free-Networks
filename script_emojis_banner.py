import os

assets_dir = "assets"
os.makedirs(assets_dir, exist_ok=True)

banner_svg = """<svg width="800" height="200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#4facfe;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#00f2fe;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#grad1)" rx="15" />
  <text x="50%" y="45%" font-family="Arial, sans-serif" font-size="40" font-weight="bold" fill="#ffffff" text-anchor="middle" dominant-baseline="middle">
    Awesome Normalization-Free Networks
  </text>
  <text x="50%" y="65%" font-family="Arial, sans-serif" font-size="20" fill="#f0f0f0" text-anchor="middle" dominant-baseline="middle">
    Zero BatchNorm, Maximum Performance
  </text>
  <circle cx="50" cy="100" r="10" fill="#ffffff">
    <animate attributeName="cx" values="50;750;50" dur="5s" repeatCount="indefinite" />
  </circle>
  <circle cx="750" cy="150" r="15" fill="#ffffff" opacity="0.5">
    <animate attributeName="cy" values="150;50;150" dur="3s" repeatCount="indefinite" />
  </circle>
</svg>"""

with open(os.path.join(assets_dir, 'banner.svg'), 'w', encoding='utf-8') as f:
    f.write(banner_svg)

with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Add banner and emojis
content = content.replace("# Awesome-Normalization-Free-Networks", "# 🚀 Awesome-Normalization-Free-Networks\n\n![Banner](assets/banner.svg)\n")
content = content.replace("## Normalization-Free Networks (NFNets)", "## 🌟 Normalization-Free Networks (NFNets)")
content = content.replace("## 1. The Macro Chronological Evolution", "## 🕰️ 1. The Macro Chronological Evolution")
content = content.replace("## 2. Core Functional & Algorithmic Variants", "## ⚙️ 2. Core Functional & Algorithmic Variants")
content = content.replace("## 3. The NFNet Layer Execution Matrix", "## 📊 3. The NFNet Layer Execution Matrix")
content = content.replace("## 4. Production Engineering Challenges & Cluster Solutions", "## 🏗️ 4. Production Engineering Challenges & Cluster Solutions")
content = content.replace("## 5. Frontier Real-World AI Industrial Applications", "## 🌍 5. Frontier Real-World AI Industrial Applications")
content = content.replace("## References", "## 📚 References")
content = content.replace("**Follow-Up Options Matrix:**", "## 🔮 Follow-Up Options Matrix:")

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(content)
