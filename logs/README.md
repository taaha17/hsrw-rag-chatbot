# Zero - Log Files

This directory contains log files for debugging and monitoring the chatbot.

## Log Files

### `zero_chat.log`
- **Purpose**: User conversations and interactions
- **Contains**: User queries, detected intents, bot responses
- **Use for**: Analyzing user behavior, popular queries, conversation flow

### `zero_debug.log`
- **Purpose**: System events, errors, and debugging information
- **Contains**: Startup info, errors, warnings, performance metrics
- **Use for**: Troubleshooting issues, system monitoring

### `zero_prompts.log`
- **Purpose**: LLM prompts and responses for refinement
- **Contains**: Full system prompts sent to Ollama, LLM responses, context used
- **Use for**: Refining prompts, improving response quality, understanding LLM behavior

## Log Rotation

All log files automatically rotate when they reach 10MB.
- Maximum 5 backup files are kept per log type
- Older backups are automatically deleted
- Files are UTF-8 encoded for international character support

## Privacy Note

Log files may contain personal conversations. Keep them secure and don't share publicly.
Add `logs/*.log` to `.gitignore` to prevent accidental commits.

## Analyzing Logs

```bash
# View recent chat conversations
tail -n 50 logs/zero_chat.log

# Watch errors in real-time
tail -f logs/zero_debug.log | grep ERROR

# Search for specific user queries
grep "User: " logs/zero_chat.log

# Check prompt engineering
less logs/zero_prompts.log
```

## Troubleshooting

If logs aren't being created:
1. Check that the `logs/` directory exists
2. Verify write permissions
3. Check `config.py` for logger initialization
4. Ensure `setup_logging()` is called at import

---

**Last Updated**: December 10, 2025
**Log Format**: UTC timestamp | logger name | level | message
