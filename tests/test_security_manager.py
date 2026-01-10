import pytest
from unittest.mock import MagicMock
from app.core.security import SecurityManager

def test_check_privacy_settings_default():
    """Test standard project returns default settings (False)."""
    mock_db = MagicMock()
    # Mock cursor to return nothing or raise error to trigger default, 
    # but the current implementation doesn't actually query DB yet, just checks ID string.
    
    manager = SecurityManager(mock_db)
    settings = manager.check_privacy_settings("standard-project-123")
    
    assert settings["do_not_train"] is False
    assert settings["sanitize_logs"] is False

def test_check_privacy_settings_secure_project():
    """Test project with '-secure' suffix triggers high privacy."""
    mock_db = MagicMock()
    manager = SecurityManager(mock_db)
    
    settings = manager.check_privacy_settings("my-secret-project-secure")
    
    assert settings["do_not_train"] is True
    assert settings["sanitize_logs"] is True

def test_fail_safe_on_error():
    """Test that manager returns strict privacy settings on DB error."""
    # We can simulate error by passing an object that raises exception on attribute access
    # or by modifying the method to raise exception.
    # Since the method catches exception, we need to make the "try" block fail.
    
    # The current implementation checks string first. To trigger Exception, 
    # we can pass a non-string project_id which might raise AttributeError on .endswith
    
    mock_db = MagicMock()
    manager = SecurityManager(mock_db)
    
    # Passing None or int might cause .endswith to fail
    settings = manager.check_privacy_settings(None) 
    
    # Should default to secure
    assert settings["do_not_train"] is True
    assert settings["sanitize_logs"] is True
