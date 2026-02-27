

claude --resume

## How to pick up a claude code session on another computer

### From existing session

Just type the following:

`/rc`

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




