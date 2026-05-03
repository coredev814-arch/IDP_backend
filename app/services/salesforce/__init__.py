"""Salesforce integration package.

Provides a client for reading case context, downloading PDFs, and (later)
writing IDP findings back to Salesforce.
"""
from app.services.salesforce.client import SalesforceClient

__all__ = ["SalesforceClient"]
