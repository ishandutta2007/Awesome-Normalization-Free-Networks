import os

with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

target = '<a href="https://discord.gg/jc4xtF58Ve"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord" /></a>'
badge_right = ' <a href="https://github.com/ishandutta2007"><img alt="GitHub followers" src="https://img.shields.io/github/followers/ishandutta2007?label=Follow" /></a>'

content = content.replace(target, target + badge_right)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(content)
