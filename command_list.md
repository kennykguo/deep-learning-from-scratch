source venv/bin/activate
deactivate

Recursively delete .DS_Store from here and now ...
find . -name '.DS_Store' -type f -delete

find . -type d \( -name "__pycache__" -o -name ".ipynb_checkpoints" \) -exec rm -rf {} +

Remove git cache:
git rm -r --cached .
git add .
git commit -am 'git cache cleared'
git push

Benchmark git commit hash
d90aaebbc050e643bc85a39193957548887c3940

find . -name "*Zone.Identifier" -type f -delete