mode: command
mode: user.dictation_command
-

# Shows the list of available prompts
{user.model} help$: user.gpt_help()

# Runs a model prompt on the selected text or other modelSource and performs modelAction on the output.
#   Example: `model paste make my email more tactful` -> Makes the selected text more tactful and pastes in place
#   Example: `model paste with clip address email to alice instead of bob` -> Rewrites the copied text and pastes it
#   Example: `model show what is the meaning of life` -> Shows the meaning of life in an overlay
#   Example: `model and show distill that to a number` -> (Following the previous prompt) shows the distilled meaning of life as a number in an overlay
#   Example: `four o mini paste make my email more tactful` -> Makes the selected text more tactful using gpt-4o-mini model
{user.model} [{user.modelThread}] {user.modelAction} [with <user.modelSource>] <user.prose>$:
    user.gpt_apply_prompt(prose, model, modelThread or "", modelSource or "", modelAction)

# Same as above, except using a saved prompt. Allows for alternative terse syntax for providing modelSource after the prompt.
#   Example: `model paste below fix grammar` -> Fixes the grammar of the selected text and pastes below
#   Example: `model paste explain` -> Explains the selected text and pastes in place
#   Example: `model show browser fix grammar clip` -> Fixes the grammar of the text on the clipboard and opens in browser`
{user.model} [{user.modelThread}] {user.modelAction} ([with <user.modelSource>] <user.modelSimplePrompt> | <user.modelSimplePrompt> <user.modelSource>)$:
    user.gpt_apply_prompt(modelSimplePrompt, model, modelThread or "", modelSource or "", modelAction)

# Select the last GPT response so you can edit it further
{user.model} take response: user.gpt_select_last()

# Applies an arbitrary prompt from the clipboard to selected text and pastes the result.
# Useful for applying complex/custom prompts that need to be drafted in a text editor.
{user.model} [{user.modelThread}] apply [from] clip$:
    prompt = clip.text()
    text = edit.selected_text()
    result = user.gpt_apply_prompt(prompt, model, modelThread or "", text)
    user.paste(result)

# Reformat the last dictation with additional context or formatting instructions
{user.model} [{user.modelThread}] [nope] that was <user.text>$:
    result = user.gpt_reformat_last(text, model, modelThread or "")
    user.paste(result)

# Enable debug logging so you can more details about messages being sent
{user.model} start debug: user.gpt_start_debug()

# Disable debug logging
{user.model} stop debug: user.gpt_stop_debug()
