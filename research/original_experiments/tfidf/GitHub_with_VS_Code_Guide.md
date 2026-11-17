# Publishing the project to GitHub from VS Code (beginner guide)

**Concepts in one minute**
- **Git** is version control. It keeps a history of snapshots of a folder on your Mac.
- **GitHub** is a website that hosts that folder online. It is the "repository" (repo) you will submit.
- A **commit** is one saved snapshot. You add a short message describing what changed.
- **Push** uploads your commits to GitHub. **Pull** downloads changes your teammates made.
- VS Code has buttons for all of this, so you won't need the command line.

---

## Part 1: One-time setup (≈10 min)
1. **Create a GitHub account** at github.com. Using your MIT email lets you get free student perks at education.github.com.
2. **Install Git.** Open the Terminal app and type `git --version`. If macOS offers to install "command line developer tools", click **Install**.
3. **Tell Git who you are.** In Terminal, paste these two lines:
   ```
   git config --global user.name "Chelsea Cheng"
   git config --global user.email "your-github-email@example.com"
   ```
4. **Sign in to GitHub inside VS Code.** Click the **Accounts** icon (person, bottom-left) and choose **Sign in with GitHub to use GitHub Pull Requests…**. A browser window opens; approve it.

## Part 2: Open the project and check it runs
1. VS Code → **File → Open Folder…** → choose `Mid term project/sole-smart-actions-intent`.
2. Open a terminal inside VS Code with **Terminal → New Terminal**, then run:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python src/run_experiments.py
   python src/predict.py "Send $50 to Jone"
   ```
   If VS Code asks *"Select interpreter"*, choose the one that shows `.venv`.

## Part 3: Publish to GitHub (first time)
1. Click the **Source Control** icon in the left bar. It looks like a branching line (shortcut ⌃⇧G).
2. Click **Publish to GitHub**.
3. Choose **Private repository** for now. You can switch it to public later.
4. VS Code lists the files to include. Keep them all checked and click **OK**. The `.gitignore` file already excludes `.venv` and temporary files.
5. When it finishes, click **Open on GitHub**. Your README is now the repo's front page.

> Before you make the repo public, confirm with your team (and Sole, if this is their data) that the dataset can be shared. It is synthetic, but the decision belongs to the data owner.

## Part 4: Everyday workflow after a change
1. Edit a file and save it. The Source Control icon shows a number (the count of changed files).
2. In the Source Control panel, type a message such as `Add confusion matrix to README`.
3. Click **✓ Commit**. If VS Code asks whether to stage all changes, answer **Yes**.
4. Click **Sync Changes** (↑↓). This pushes your commit and pulls your teammates' commits.

## Part 5: Working with teammates
- On GitHub, open the repo → **Settings → Collaborators → Add people**. Invite them by GitHub username.
- Teammates get a copy with VS Code → **Clone Git Repository…** → paste the repo URL.
- Click **Sync** before you start working and again when you finish. That avoids most conflicts.
- If VS Code shows a **merge conflict**, the file highlights both versions. Click **Accept Current**, **Accept Incoming**, or **Accept Both**, save, then commit.

## Submitting
Copy the repo URL (e.g., `https://github.com/<your-username>/sole-smart-actions-intent`) into the course submission. If the repo is private, add the instructors as collaborators, or make it public after your team agrees.
