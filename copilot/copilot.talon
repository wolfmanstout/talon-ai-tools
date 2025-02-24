app: vscode
not tag: user.codeium
-
pilot jest: user.vscode("editor.action.inlineSuggest.trigger")
pilot next: user.vscode("editor.action.inlineSuggest.showNext")
pilot (previous | last): user.vscode("editor.action.inlineSuggest.showPrevious")
pilot yes: user.vscode("editor.action.inlineSuggest.commit")
pilot yes word: user.vscode("editor.action.inlineSuggest.acceptNextWord")
pilot nope: user.vscode("editor.action.inlineSuggest.undo")
pilot cancel: user.vscode("editor.action.inlineSuggest.hide")
pilot chat last: user.vscode("workbench.action.chat.previousCodeBlock")
pilot chat next: user.vscode("workbench.action.chat.nextCodeBlock")
pilot chat bring <user.cursorless_ordinal_or_last> to new file:
    user.copilot_focus_code_block(cursorless_ordinal_or_last)
    user.vscode("workbench.action.chat.insertIntoNewFile")
pilot chat copy <user.cursorless_ordinal_or_last>:
    user.copilot_focus_code_block(cursorless_ordinal_or_last)
    edit.copy()
pilot chat bring <user.cursorless_ordinal_or_last>:
    user.copilot_bring_code_block(cursorless_ordinal_or_last)
pilot chat bring <user.cursorless_ordinal_or_last> {user.makeshift_destination} <user.cursorless_target>:
    user.cursorless_command(makeshift_destination, cursorless_target)
    user.copilot_bring_code_block(cursorless_ordinal_or_last)
pilot {user.copilot_slash_command} <user.cursorless_target> [to <user.prose>]$:
    user.cursorless_command("setSelection", cursorless_target)
    user.copilot_inline_chat(copilot_slash_command or "", prose or "")
pilot make [<user.prose>]: user.copilot_inline_chat("", prose or "")
pilot chat new: user.vscode("workbench.action.chat.newChat")
pilot edit new: user.vscode("workbench.action.chat.newEditSession")
pilot edit attach: user.vscode("github.copilot.edits.attachFile")
pilot chat attach: user.vscode("github.copilot.chat.attachFile")
[pilot] edit next: user.vscode("chatEditor.action.navigateNext")
[pilot] edit last: user.vscode("chatEditor.action.navigatePrevious")
pilot edit accept: user.vscode("chatEditor.action.acceptHunk")
pilot edit accept file: user.vscode("chatEditor.action.accept")
pilot edit accept all: user.vscode("chatEditing.acceptAllFiles")
pilot edit discard: user.vscode("chatEditor.action.undoHunk")
pilot edit discard file: user.vscode("chatEditor.action.reject")
pilot edit discard all: user.vscode("chatEditing.discardAllFiles")
pilot edit diff: user.vscode("chatEditor.action.diffHunk")
