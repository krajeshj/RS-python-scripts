import os
import re

dir_path = 'output'
files = [f for f in os.listdir(dir_path) if f.endswith('.html')]

new_link = '            <a href="sprints.html" class="nav-link">Sprints ??</a>\n'

for filename in files:
    path = os.path.join(dir_path, filename)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'sprints.html' in content:
        continue
        
    # Find the nav-links div
    nav_pattern = re.compile(r'(<div class="nav-links">.*?)(</div>)', re.DOTALL)
    match = nav_pattern.search(content)
    if match:
        nav_inner = match.group(1)
        closing = match.group(2)
        
        # If the file is sprints.html, make it active
        if filename == 'sprints.html':
            # sprints.html already has its own nav in the template I wrote earlier
            # but I'll update it to match the others if needed.
            # Actually my sprints.html template used a different nav style.
            # Let me just check index.html's nav style first.
            continue
            
        new_nav = nav_inner + new_link + closing
        new_content = content[:match.start()] + new_nav + content[match.end():]
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filename}")
