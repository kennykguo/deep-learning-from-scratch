source venv/bin/activate
deactivate

Recursively delete .DS_Store from here and now ...
find . -name '.DS_Store' -type f -delete

Remove git cache:
git rm -r --cached .
git add .
git commit -am 'git cache cleared'
git push