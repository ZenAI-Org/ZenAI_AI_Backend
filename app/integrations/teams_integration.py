import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

class TeamsIntegration:
    """
    Integration with Microsoft Teams via Incoming Webhooks.
    """
    
    def __init__(self):
        self.webhook_url = os.getenv("TEAMS_WEBHOOK_URL")
        
    def send_notification(self, title: str, message: str, color: str = "0076D7") -> bool:
        """
        Send a card notification to Teams.
        """
        if not self.webhook_url:
            logger.warning("TEAMS_WEBHOOK_URL not set")
            return False
            
        try:
            # Teams connector card format
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": color,
                "summary": title,
                "sections": [{
                    "activityTitle": title,
                    "activitySubtitle": "AI Project Manager Agent",
                    "text": message
                }]
            }
            
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status() # Raises error for 4xx/5xx
            
            # Teams returns "1" on success usually
            if response.text == "1" or response.status_code == 200:
                logger.info("Sent Teams notification")
                return True
            else:
                logger.warning(f"Teams webhook returned unexpected response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Teams notification: {e}")
            return False
