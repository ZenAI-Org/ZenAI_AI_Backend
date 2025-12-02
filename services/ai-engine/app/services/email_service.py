import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

class EmailService:
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.from_email = os.getenv("SMTP_USER")
        
        if not self.smtp_user or not self.smtp_password:
            print("[WARNING] Email credentials not configured")
        else:
            print(f"[INFO] Email service configured for: {self.smtp_user}")
    
    def send_email(
        self,
        to_emails: List[str],
        subject: str,
        body_text: str,
        body_html: Optional[str] = None
    ) -> bool:
        """Send an email"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = subject
            
            # Add plain text version
            part1 = MIMEText(body_text, 'plain')
            msg.attach(part1)
            
            # Add HTML version if provided
            if body_html:
                part2 = MIMEText(body_html, 'html')
                msg.attach(part2)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            print(f"[INFO] Email sent successfully to {', '.join(to_emails)}")
            return True
        
        except Exception as e:
            print(f"[ERROR] Failed to send email: {e}")
            return False
    
    def send_daily_report(self, report_markdown: str, to_email: str) -> bool:
        """Send daily report email"""
        
        # Convert markdown to HTML
        html_body = self._markdown_to_html(report_markdown)
        
        return self.send_email(
            to_emails=[to_email],
            subject="Daily Project Report - ZenAI",
            body_text=report_markdown,
            body_html=html_body
        )
    
    def send_overdue_alert(self, task_title: str, assignee: str, days_overdue: int, task_url: str, to_email: str) -> bool:
        """Send overdue task alert"""
        
        subject = f" Overdue Task Alert: {task_title}"
        
        body_text = f"""
Hi {assignee},

This is a reminder that your task is now overdue:

Task: {task_title}
Days Overdue: {days_overdue}
Task URL: {task_url}

Please update the task status or reach out if you need help!

Best regards,
ZenAI Project Manager
"""
        
        body_html = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <div style="background-color: #d32f2f; color: white; padding: 20px; text-align: center;">
        <h1 style="margin: 0;"> Overdue Task Alert</h1>
    </div>
    <div style="padding: 20px; background-color: #f5f5f5;">
        <p>Hi <strong>{assignee}</strong>,</p>
        <p>This is a reminder that your task is now overdue:</p>
        <div style="background-color: #ffebee; padding: 20px; border-left: 5px solid #d32f2f; margin: 20px 0;">
            <p style="margin: 5px 0;"><strong>Task:</strong> {task_title}</p>
            <p style="margin: 5px 0;"><strong>Days Overdue:</strong> {days_overdue}</p>
        </div>
        <p style="text-align: center;">
            <a href="{task_url}" style="background-color: #d32f2f; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block;">View Task in Notion</a>
        </p>
        <p>Please update the task status or reach out if you need help!</p>
        <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
        <p style="color: #666; font-size: 12px;">
            Best regards,<br>
            <strong>ZenAI Project Manager</strong>
        </p>
    </div>
</body>
</html>
"""
        
        return self.send_email(
            to_emails=[to_email],
            subject=subject,
            body_text=body_text,
            body_html=body_html
        )
    
    def send_deadline_reminder(self, task_title: str, assignee: str, days_until_due: int, task_url: str, to_email: str) -> bool:
        """Send deadline reminder"""
        
        subject = f" Deadline Reminder: {task_title}"
        
        body_text = f"""
Hi {assignee},

Friendly reminder that your task is coming up soon:

Task: {task_title}
Due in: {days_until_due} days
Task URL: {task_url}

Keep up the great work!

Best regards,
ZenAI Project Manager
"""
        
        body_html = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
    <div style="background-color: #ff9800; color: white; padding: 20px; text-align: center;">
        <h1 style="margin: 0;"> Deadline Reminder</h1>
    </div>
    <div style="padding: 20px; background-color: #f5f5f5;">
        <p>Hi <strong>{assignee}</strong>,</p>
        <p>Friendly reminder that your task is coming up soon:</p>
        <div style="background-color: #fff3e0; padding: 20px; border-left: 5px solid #ff9800; margin: 20px 0;">
            <p style="margin: 5px 0;"><strong>Task:</strong> {task_title}</p>
            <p style="margin: 5px 0;"><strong>Due in:</strong> {days_until_due} days</p>
        </div>
        <p style="text-align: center;">
            <a href="{task_url}" style="background-color: #ff9800; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block;">View Task in Notion</a>
        </p>
        <p>Keep up the great work!</p>
        <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
        <p style="color: #666; font-size: 12px;">
            Best regards,<br>
            <strong>ZenAI Project Manager</strong>
        </p>
    </div>
</body>
</html>
"""
        
        return self.send_email(
            to_emails=[to_email],
            subject=subject,
            body_text=body_text,
            body_html=body_html
        )
    
    def _markdown_to_html(self, markdown: str) -> str:
        """Basic markdown to HTML conversion"""
        import re
        
        html = markdown
        
        # Escape HTML characters
        html = html.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Headers
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        
        # Bold
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        
        # Lists
        lines = html.split('\n')
        in_list = False
        result = []
        
        for line in lines:
            if line.strip().startswith('- '):
                if not in_list:
                    result.append('<ul>')
                    in_list = True
                result.append(f'<li>{line.strip()[2:]}</li>')
            else:
                if in_list:
                    result.append('</ul>')
                    in_list = False
                if line.strip():
                    result.append(f'<p>{line}</p>')
                else:
                    result.append('<br>')
        
        if in_list:
            result.append('</ul>')
        
        # Wrap in basic HTML
        final_html = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5;">
    <div style="background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        {''.join(result)}
    </div>
</body>
</html>
"""
        return final_html