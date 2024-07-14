# bikini-bottom
Scripts to manipulate cloudvolumes.

[Inspired by Patrick Star](https://www.youtube.com/watch?v=A_pTGhOyPbw), the tools here take mesh or image data (often from a cloudvolume), do something with it, and push it somewhere else (often another cloudvolume).


### Installation
As is always the case in python, consider making a virtual environment (using your preference of conda, virtualenv, or virtualenvwrapper) before installing.

**Option 1:** `pip install` from PyPI:

    pip install bikinibottom

**Option 2:** `pip install` the latest version from GitHub:
    
    pip install git+https://github.com/jasper-tms/bikini-bottom.git

**Option 3:** First `git clone` this repo and then `pip install` it from your clone:

    cd ~/repos  # Or wherever on your computer you want to download this code to
    git clone https://github.com/jasper-tms/bikini-bottom.git
    cd bikini-bottom 
    pip install .

**After installing,** you can import this package in python using `import bikinibottom` (not `import bikini-bottom`!)
