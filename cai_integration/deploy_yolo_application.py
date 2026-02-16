#!/usr/bin/env python3
"""
Deploy YOLO Detection API as CAI Application.

This script creates/updates a CAI Application to serve the trained YOLO model
as a persistent API with exposed endpoints.

Usage:
    python cai_integration/deploy_yolo_application.py

Environment Variables:
    CAI_API_KEY: Cloudera AI API key (required)
    CAI_DOMAIN: CAI domain URL (required)
    CAI_PROJECT_NAME: Project name (default: current project)
    MODEL_PATH: Path to trained model (default: latest from runs/detect/)
    APP_SUBDOMAIN: Custom subdomain for application (optional)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import requests


# Color output
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_status(message: str, status: str = "info"):
    """Print colored status message."""
    colors = {
        "info": Colors.BLUE,
        "success": Colors.GREEN,
        "warning": Colors.YELLOW,
        "error": Colors.RED
    }
    color = colors.get(status, Colors.NC)
    print(f"{color}{message}{Colors.NC}")


def find_latest_model() -> Optional[Path]:
    """Find the latest trained YOLO model."""
    runs_dir = Path("runs/detect")
    
    if not runs_dir.exists():
        return None
    
    # Look for subdirectories with best.pt
    model_dirs = []
    for subdir in runs_dir.iterdir():
        if subdir.is_dir():
            model_path = subdir / "weights" / "best.pt"
            if model_path.exists():
                model_dirs.append((subdir, model_path.stat().st_mtime))
    
    if not model_dirs:
        return None
    
    # Return most recent
    model_dirs.sort(key=lambda x: x[1], reverse=True)
    latest_dir = model_dirs[0][0]
    return latest_dir / "weights" / "best.pt"


def get_cai_client(api_key: str, domain: str) -> requests.Session:
    """Create authenticated CAI API client."""
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    })
    return session


def get_project_id(client: requests.Session, domain: str, project_name: Optional[str] = None) -> str:
    """Get CAI project ID."""
    url = f"{domain}/api/v2/projects"
    response = client.get(url)
    response.raise_for_status()
    
    projects = response.json()
    
    if project_name:
        # Find by name
        for project in projects:
            if project.get("name") == project_name:
                return project["id"]
        raise ValueError(f"Project '{project_name}' not found")
    
    # Use first project
    if not projects:
        raise ValueError("No projects found")
    
    return projects[0]["id"]


def create_or_update_application(
    client: requests.Session,
    domain: str,
    project_id: str,
    model_path: str,
    subdomain: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create or update CAI Application for YOLO API.
    
    Returns application details including endpoint URL.
    """
    # Application configuration
    app_name = "xray-yolo-detection-api"
    app_config = {
        "name": app_name,
        "description": "YOLO-based X-ray baggage threat detection API",
        "subdomain": subdomain or "xray-yolo-api",
        "script": "cai_integration/launch_yolo_application.sh",
        "kernel": "python3",
        "cpu": 4,
        "memory": 16,
        "gpu": 1,
        "runtime_identifier": "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-jupyterlab-python3.10-cuda:2025.09.1-b5",
        "environment": {
            "MODEL_PATH": model_path,
            "BACKEND": "ultralytics",
            "CONF_THRESHOLD": "0.25",
            "IOU_THRESHOLD": "0.45",
            "DEVICE": "0",
            "HOST": "0.0.0.0",
            "PORT": "8080"  # CAI Applications use port 8080
        }
    }
    
    # Check if application already exists
    list_url = f"{domain}/api/v2/projects/{project_id}/applications"
    response = client.get(list_url)
    response.raise_for_status()
    
    existing_apps = response.json()
    existing_app = None
    
    for app in existing_apps:
        if app.get("name") == app_name:
            existing_app = app
            break
    
    if existing_app:
        # Update existing application
        print_status(f"Updating existing application: {app_name}", "info")
        update_url = f"{domain}/api/v2/projects/{project_id}/applications/{existing_app['id']}"
        response = client.patch(update_url, json=app_config)
        response.raise_for_status()
        
        # Restart application
        restart_url = f"{update_url}/restart"
        response = client.post(restart_url)
        response.raise_for_status()
        
        return existing_app
    
    else:
        # Create new application
        print_status(f"Creating new application: {app_name}", "info")
        response = client.post(list_url, json=app_config)
        response.raise_for_status()
        
        return response.json()


def main():
    parser = argparse.ArgumentParser(
        description="Deploy YOLO Detection API as CAI Application"
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv('CAI_API_KEY'),
        help='CAI API key (or set CAI_API_KEY env var)'
    )
    parser.add_argument(
        '--domain',
        type=str,
        default=os.getenv('CAI_DOMAIN'),
        help='CAI domain URL (or set CAI_DOMAIN env var)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default=os.getenv('CAI_PROJECT_NAME'),
        help='Project name (or set CAI_PROJECT_NAME env var)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=os.getenv('MODEL_PATH'),
        help='Path to trained model (or set MODEL_PATH env var, auto-detects if not set)'
    )
    parser.add_argument(
        '--subdomain',
        type=str,
        default=os.getenv('APP_SUBDOMAIN'),
        help='Custom subdomain for application (optional)'
    )
    
    args = parser.parse_args()
    
    print_status("=" * 60, "info")
    print_status("Deploy YOLO Detection API as CAI Application", "info")
    print_status("=" * 60, "info")
    print()
    
    # Validate required parameters
    if not args.api_key:
        print_status("Error: CAI_API_KEY not set", "error")
        print_status("Set via --api-key or CAI_API_KEY environment variable", "warning")
        return 1
    
    if not args.domain:
        print_status("Error: CAI_DOMAIN not set", "error")
        print_status("Set via --domain or CAI_DOMAIN environment variable", "warning")
        print_status("Example: https://ml-xxx.cloudera.site", "info")
        return 1
    
    # Find model
    model_path = args.model
    if not model_path:
        print_status("No model specified, searching for latest trained model...", "info")
        model_path = find_latest_model()
        
        if not model_path:
            print_status("Error: No trained model found in runs/detect/", "error")
            print_status("Train a model first or specify --model", "warning")
            return 1
        
        print_status(f"Found latest model: {model_path}", "success")
    else:
        model_path = Path(model_path)
        if not model_path.exists():
            print_status(f"Error: Model not found: {model_path}", "error")
            return 1
    
    model_path_str = str(model_path)
    
    print()
    print_status("Configuration:", "info")
    print(f"  Domain:     {args.domain}")
    print(f"  Project:    {args.project or '(auto-detect)'}")
    print(f"  Model:      {model_path_str}")
    print(f"  Subdomain:  {args.subdomain or 'xray-yolo-api (default)'}")
    print()
    
    try:
        # Create API client
        print_status("Authenticating with CAI...", "info")
        client = get_cai_client(args.api_key, args.domain)
        
        # Get project ID
        print_status("Getting project information...", "info")
        project_id = get_project_id(client, args.domain, args.project)
        print_status(f"✓ Project ID: {project_id}", "success")
        
        # Deploy application
        print_status("Deploying YOLO API application...", "info")
        app = create_or_update_application(
            client,
            args.domain,
            project_id,
            model_path_str,
            args.subdomain
        )
        
        print()
        print_status("=" * 60, "success")
        print_status("✅ Application Deployed Successfully!", "success")
        print_status("=" * 60, "success")
        print()
        
        # Print application details
        subdomain = args.subdomain or "xray-yolo-api"
        app_url = f"{args.domain}/applications/{subdomain}"
        
        print_status("Application Details:", "info")
        print(f"  Name:        xray-yolo-detection-api")
        print(f"  Application ID: {app.get('id', 'N/A')}")
        print(f"  Status:      Starting (wait 1-2 minutes)")
        print()
        
        print_status("Access Endpoints:", "info")
        print(f"  Application UI:   {app_url}")
        print(f"  Health Check:     {app_url}/health")
        print(f"  API Docs:         {app_url}/docs")
        print(f"  OpenAI API:       {app_url}/v1/chat/completions")
        print(f"  Direct Detection: {app_url}/v1/detect")
        print()
        
        print_status("Testing the API:", "info")
        print(f"  curl {app_url}/health")
        print()
        print(f"  curl -X POST {app_url}/v1/detect \\")
        print(f"    -F 'file=@path/to/xray.jpg'")
        print()
        
        print_status("Next Steps:", "info")
        print("  1. Wait 1-2 minutes for application to start")
        print("  2. Check application status in CAI UI")
        print(f"  3. Test health endpoint: curl {app_url}/health")
        print(f"  4. View API docs: {app_url}/docs")
        print()
        
        return 0
        
    except requests.HTTPError as e:
        print_status(f"HTTP Error: {e.response.status_code}", "error")
        print_status(f"Response: {e.response.text}", "error")
        return 1
    
    except Exception as e:
        print_status(f"Error: {str(e)}", "error")
        return 1


if __name__ == '__main__':
    sys.exit(main())
