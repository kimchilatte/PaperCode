# Issue

If there is some problems first left the issue

# Contribution guidelines

1. Fork the repository
2. download to your local computer
    ```
    $ git clone https://github.com/`YOUR_GITHUB_NAME`/PaperCode.git
    $ cd PaperCode
    ```
3. Setup an upstream
    ```
    $ git remote add upstream https://github.com/kimchilatte/PaperCode.git
    ```
    You can update if there is some updated form the orignal repository
    ```
    $ git pull upstream master && git push origin master
    ```
4. Make a branch

    Do not push your codes directly to the master branch. Create a branch.

    the branch name should be paper + `issue number`, e.x) paper3

    ```
    $ git checkout -b `paper + issue number` -t origin/master
    ```
5. Commit

    Set the config

    ```
    $ git config --global user.name "`username`"
    $ git config --global user.email "`user@email`" 
    ```

    After changing the codes, add & commit. 
    - please do not add your datas in this repository
    - all your code should be in the single folder

    ```
    $ git add `changes`
    $ git commit -m "`write the message link to paper issue`
    ```

6. (optional) Rebase the branch

    ```
    $ git fetch upstream
    $ git rebase upstream/master
    ```

7. Push

    ```
    $ git push -u origin `paper + issue number`
    ```

8. Create Pull Request on the web and see the codes