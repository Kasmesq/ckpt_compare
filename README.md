## setup 
(One-time) Fix python alias to avoid python3.9: command not found
```bash
cd ~

# Comment out any python -> python3.9 aliases in ~/.bashrc
sed -i 's/^alias python=python3\.9/# alias python=python3.9/' ~/.bashrc
sed -i 's/^alias python3=python3\.9/# alias python3=python3.9/' ~/.bashrc

# Reload shell config
source ~/.bashrc
hash -r

# Check that python now comes from conda, not an alias
type -a python
python --version
```bash
