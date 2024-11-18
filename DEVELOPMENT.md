
### Pre-Commit Hooks

The hooks are required for the team. Although skipping this step does not prevent you from making a commit, this step is critical in maintaining a uniform coding style across developers.

From now on, whenever you make a new commit, you should see logs like this in your terminal:

```
Check Yaml...............................................................Passed
Fix End of Files.........................................................Passed
Trim Trailing Whitespace.................................................Passed
black....................................................................Passed
```

If you see [Black](https://github.com/psf/black) failed and modified files:

```
Check Yaml...............................................................Passed
Fix End of Files.........................................................Passed
Trim Trailing Whitespace.................................................Passed
black....................................................................Failed
- hook id: black
- files were modified by this hook

reformatted path/to/file.py
All done! ✨ 🍰 ✨
1 file reformatted.
```

Then `git add` the auto-modified files and retry your `git commit` command.

Please report a bug if you do not see such messages.
