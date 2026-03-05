

claude --resume

## Remote Contrl

How to pick up a claude code session on another computer

### From existing session

Just type the following:

`/rc`



### Joining RC Session

Go to https://claude.ai/code

From there you can enable notifications so you know when to check in.


## Renaming

You can rename your sessions with `/rename` to pick them up later:

`/rename my new name`

then you can restart it with `claude --resume "my new name"`

I would recommend snake case so you don't have to use the quotes:

`/rename ny_new_name`

`claude --resume my_new_name`


## Using Vim Mode

You can go into `~/.claude.json` and add `"editorMode": "vim",`

See my [vim notes](https://jss367.github.io/vim-notes.html)/


## Clearing context

Useful if you want to switch topics.

`/clear`

## Support files

The main support file you use is CLAUDE.md. If it's at ~/CLAUDE.md, it will be loaded into every context. You can also put one in your working directory.

You can also use SKILL.md. For these, only the description is loaded. Then Claude checks whether the description is loaded 






