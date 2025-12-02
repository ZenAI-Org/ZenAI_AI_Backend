"""
Authentication and authorization middleware for AI workflows.

Provides:
- User authentication validation
- Role-based access control (RBAC)
- Project-level permission checks
- Token validation
"""

from fastapi import Request, HTTPException, status
from typing import Optional, Dict, Any
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class AuthContext:
    """Context object for authenticated requests."""
    
    def __init__(
        self,
        user_id: str,
        user_role: str,
        project_id: Optional[str] = None,
        permissions: Optional[Dict[str, bool]] = None,
    ):
        """
        Initialize auth context.
        
        Args:
            user_id: User ID
            user_role: User role (admin, manager, member)
            project_id: Optional project ID
            permissions: Optional permissions dictionary
        """
        self.user_id = user_id
        self.user_role = user_role
        self.project_id = project_id
        self.permissions = permissions or {}
        self.authenticated_at = datetime.now()
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return self.permissions.get(permission, False)
    
    def is_admin(self) -> bool:
        """Check if user is admin."""
        return self.user_role == "admin"
    
    def is_manager(self) -> bool:
        """Check if user is manager."""
        return self.user_role in ["admin", "manager"]
    
    def can_access_project(self, project_id: str) -> bool:
        """Check if user can access project."""
        if self.is_admin():
            return True
        return self.project_id == project_id


class AuthenticationMiddleware:
    """
    Middleware for request authentication.
    
    Validates bearer tokens and extracts user information.
    """
    
    def __init__(self, app):
        """Initialize middleware."""
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """Process request using ASGI interface."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Skip auth for health check and root endpoints
        if request.url.path in ["/", "/health", "/api/health"]:
            await self.app(scope, receive, send)
            return
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            # For now, allow requests without auth (can be made stricter)
            # In production, this should validate tokens
            scope["state"] = {"auth": AuthContext(
                user_id="anonymous",
                user_role="member",
            )}
            await self.app(scope, receive, send)
            return
        
        try:
            # Parse bearer token
            scheme, token = auth_header.split()
            
            if scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme",
                )
            
            # Validate token (in production, verify JWT signature)
            auth_context = self._validate_token(token)
            scope["state"] = {"auth": auth_context}
            
            logger.info(
                f"User authenticated: user_id={auth_context.user_id}, "
                f"role={auth_context.user_role}"
            )
        
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
            )
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed",
            )
        
        await self.app(scope, receive, send)
    
    def _validate_token(self, token: str) -> AuthContext:
        """
        Validate bearer token.
        
        In production, this should:
        1. Verify JWT signature
        2. Check token expiration
        3. Extract user claims
        
        Args:
            token: Bearer token
            
        Returns:
            AuthContext with user information
            
        Raises:
            ValueError: If token is invalid
        """
        # For development, accept any non-empty token
        # In production, implement proper JWT validation
        
        if not token or not token.strip():
            raise ValueError("Token is empty")
        
        # Mock token validation - in production use PyJWT
        # Example: jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        
        # For now, extract user info from token (mock implementation)
        # In production, decode JWT and extract claims
        
        return AuthContext(
            user_id="user_123",
            user_role="member",
            permissions={
                "read_projects": True,
                "create_tasks": True,
                "read_chat": True,
            },
        )


class AuthorizationMiddleware:
    """
    Middleware for request authorization.
    
    Validates user permissions for specific resources.
    """
    
    def __init__(self, app):
        """Initialize middleware."""
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """Process request using ASGI interface."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Get auth context from scope state
        auth = scope.get("state", {}).get("auth")
        
        if not auth:
            # No auth context, allow request to proceed
            # (will be handled by endpoint-level checks)
            await self.app(scope, receive, send)
            return
        
        # Check project-level permissions
        project_id = self._extract_project_id(request)
        
        if project_id and not auth.can_access_project(project_id):
            logger.warning(
                f"Unauthorized project access: user_id={auth.user_id}, "
                f"project_id={project_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this project",
            )
        
        await self.app(scope, receive, send)
    
    def _extract_project_id(self, request: Request) -> Optional[str]:
        """Extract project_id from request path or query params."""
        # Check path parameters
        if "project_id" in request.path_params:
            return request.path_params["project_id"]
        
        # Check query parameters
        if "project_id" in request.query_params:
            return request.query_params["project_id"]
        
        return None


def get_auth_context(request: Request) -> AuthContext:
    """
    Dependency for getting auth context from request.
    
    Args:
        request: FastAPI request
        
    Returns:
        AuthContext with user information
        
    Raises:
        HTTPException: If not authenticated
    """
    auth = getattr(request.state, "auth", None)
    
    if not auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    
    return auth


def require_role(required_role: str):
    """
    Dependency for requiring specific role.
    
    Args:
        required_role: Required role (admin, manager, member)
        
    Returns:
        Dependency function
    """
    async def check_role(auth: AuthContext = None):
        if not auth:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
            )
        
        role_hierarchy = {"admin": 3, "manager": 2, "member": 1}
        user_level = role_hierarchy.get(auth.user_role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires {required_role} role",
            )
        
        return auth
    
    return check_role


def require_permission(permission: str):
    """
    Dependency for requiring specific permission.
    
    Args:
        permission: Required permission
        
    Returns:
        Dependency function
    """
    async def check_permission(auth: AuthContext = None):
        if not auth:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
            )
        
        if not auth.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires {permission} permission",
            )
        
        return auth
    
    return check_permission
