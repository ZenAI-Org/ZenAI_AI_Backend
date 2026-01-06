import os
import logging
import requests
from requests.auth import HTTPBasicAuth
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class JiraIntegration:
    """
    Integration with Atlassian Jira Cloud REST API v3.
    """
    
    def __init__(self):
        self.base_url = os.getenv("JIRA_BASE_URL") # e.g., https://your-domain.atlassian.net
        self.email = os.getenv("JIRA_EMAIL")
        self.api_token = os.getenv("JIRA_API_Token")
        self.project_key = os.getenv("JIRA_PROJECT_KEY")
        
        self.is_configured = all([self.base_url, self.email, self.api_token, self.project_key])
        
        if self.is_configured:
            self.auth = HTTPBasicAuth(self.email, self.api_token)
            self.headers = {
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        else:
            logger.warning("Jira integration incomplete. Check JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY")

    def create_issue(self, summary: str, description: str, issue_type: str = "Task", priority: str = "Medium") -> Optional[str]:
        """
        Create a new issue in Jira.
        Returns the Issue Key (e.g., PROJ-123) or None.
        """
        if not self.is_configured:
            return None
            
        url = f"{self.base_url}/rest/api/3/issue"
        
        # Map simple priority to Jira priority IDs if needed, 
        # but for simplicity we'll just try to send standard JSON structure
        # Note: Jira API structure is very customized per instance. This is a generic "Core" structure.
        
        payload = {
            "fields": {
                "project": {
                    "key": self.project_key
                },
                "summary": summary,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {
                                    "type": "text",
                                    "text": description
                                }
                            ]
                        }
                    ]
                },
                "issuetype": {
                    "name": issue_type
                }
            }
        }
        
        try:
            response = requests.post(url, json=payload, headers=self.headers, auth=self.auth)
            
            if response.status_code == 201:
                data = response.json()
                key = data.get("key")
                logger.info(f"Created Jira issue: {key}")
                return key
            else:
                logger.error(f"Failed to create Jira issue: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Jira API exception: {e}")
            return None

    def get_issue_status(self, issue_key: str) -> Optional[str]:
        """
        Get the status of an existing issue.
        """
        if not self.is_configured:
            return None
            
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        
        try:
            response = requests.get(url, headers=self.headers, auth=self.auth)
            if response.status_code == 200:
                data = response.json()
                status = data.get("fields", {}).get("status", {}).get("name")
                return status
            return None
        except Exception as e:
            logger.error(f"Jira get status failed: {e}")
            return None
