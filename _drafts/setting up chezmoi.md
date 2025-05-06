setting up chezmoi



~/git/dotfiles main !1 ❯ chezmoi managed                                                                                                 base

README.md
git
git/gitconfig
shell
shell/profile
shell/zshrc



So even if you cloned your dotfiles repo into ~/git/dotfiles, chezmoi isn’t watching that. It’s managing its own clone of the repo, and you’re supposed to do all dotfile changes either:

In-place (e.g., edit ~/.zshrc then chezmoi add ~/.zshrc)

Or inside chezmoi cd (to modify templates or source-controlled files)




Key Principle: Chezmoi is the source of truth
You should:

Edit files in ~/.zshrc

Track them with chezmoi (which copies them into ~/.local/share/chezmoi in a Git-friendly format)

Then commit from inside chezmoi's source dir

So:
✅ ~/.zshrc — where you live-edit
✅ ~/.local/share/chezmoi/dot_zshrc — chezmoi’s Git-managed version



This also means that chezmoi is the only repo that isn't with my other repos.



After you edit a file:

* `chezmoi diff`
* `chezmoi add ~/.profile`

Then, commit and push the change:
* `chezmoi cd`
* `git add dot_profile`
* `git commit -m "Update profile"`
* `git push`



## Setting up a work computer

This is supposed to work but didn't
`chezmoi init --apply --promptBool workComputer=true`   <------------- this didn't work.

Instead,  create this file:

data:
  workComputer: true

and save it as `~/.config/chezmoi/chezmoi.yaml`


Then we have this:

```
{{ if .workComputer }}
source .work_profile
{{ end }}
```

If you want to use chezmoi-specific stuff like the example above, you'll need to rename your file to use .tmpl:
In your chezmoi source directory:
`mv ~/.local/share/chezmoi/dot_profile ~/.local/share/chezmoi/dot_profile.tmpl`

Chezmoi will now treat .profile as a template and process {{ if ... }} blocks during chezmoi apply.


So now, here's what I do:

To make a change to .profile, you should edit:

`~/.local/share/chezmoi/dot_profile.tmpl`

Then apply your changes with:

`chezmoi apply`

This updates the actual ~/.profile on disk.








