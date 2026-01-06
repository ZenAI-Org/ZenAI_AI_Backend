import os
import logging
import requests
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

class SlackIntegration:
    """
    Integration with Slack for messaging and notifications.
    Uses generic Webhooks for simplicity, or Bot Token for advanced features.
    """
    
    def __init__(self):
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.default_channel = os.getenv("SLACK_DEFAULT_CHANNEL", "#general")
        
    def send_notification(self, message: str, channel: Optional[str] = None) -> bool:
        """
        Send a simple text notification.
        """
        try:
            # 1. Try Webhook first (simplest)
            if self.webhook_url:
                payload = {"text": message}
                response = requests.post(self.webhook_url, json=payload)
                response.raise_for_status()
                logger.info("Sent Slack webhook notification")
                return True
                
            # 2. Fallback to Bot API
            elif self.bot_token:
                channel_id = channel or self.default_channel
                url = "https://slack.com/api/chat.postMessage"
                headers = {
                    "Authorization": f"Bearer {self.bot_token}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "channel": channel_id,
                    "text": message
                }
                response = requests.post(url, headers=headers, json=payload)
                data = response.json()
                if not data.get("ok"):
                    logger.error(f"Slack API error: {data.get('error')}")
                    return False
                logger.info(f"Sent Slack API message to {channel_id}")
                return True
                
            else:
                logger.warning("No Slack configuration found (SLACK_WEBHOOK_URL or SLACK_BOT_TOKEN)")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    def send_block_message(self, blocks: List[Dict], text: str = "New Notification") -> bool:
        """
        Send a structured block message (Slack's UI format).
        """
        try:
            if self.webhook_url:
                payload = {"text": text, "blocks": blocks}
                response = requests.post(self.webhook_url, json=payload)
                response.raise_for_status()
                return True
            elif self.bot_token:
                logger.warning("Block messages not yet fully implemented for Bot Token in this simple wrapper")
                return False
            return False
        except Exception as e:
            logger.error(f"Failed to send Slack block message: {e}")
            return False
