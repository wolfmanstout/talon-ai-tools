# Clipboard Format Enhancement Plan

## Overview

Add support for "clip as html" and "clip as markdown" functionality to allow users to access HTML content from the clipboard and optionally convert it to markdown using the markitdown CLI tool.

## Implementation Plan

1. Create a new list file `GPT/lists/modelFormat.talon-list`:

   ```
   list: user.modelFormat
   -
   html: html
   markdown: markdown
   ```

2. In `lib/talonSettings.py`:

   - Add a setting for markitdown path
   - Add a list definition for modelFormat
   - Update ContentSpec to include format
   - Create a new capture rule for "source as format"
   - Update the combined modelSource capture to include the new rule

3. In `lib/modelHelpers.py`:

   - Add function `get_clipboard_html()` to get HTML from clipboard
   - Add function `convert_html_to_markdown()` to use markitdown CLI

4. In `GPT/gpt.py`:
   - Update `resolve_source` to handle format when source is "clipboard"
   - Update `gpt_get_source_text` to pass format to resolve_source

This approach keeps the solution clean and extensible by using a list for formats rather than hardcoding them, and integrates with the existing pattern of "source as X" modifiers.
