"""Salesforce REST/SOAP client for the IDP audit pipeline.

Read-only for v1. Write methods (update_case_findings, mark_complete) are
deliberately not implemented yet — the pipeline produces findings text but
does not push it to Salesforce until that integration is approved.
"""
import logging
import threading
import time
from typing import Any

from simple_salesforce import Salesforce

from app.core.config import Settings

logger = logging.getLogger(__name__)


def _escape_soql(value: str) -> str:
    """Escape a value for inclusion in a SOQL string literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


# Module-level singleton — re-using the auth handshake across jobs avoids
# unnecessary Salesforce login calls (each job previously re-auth'd).
_CLIENT: "SalesforceClient | None" = None
_CLIENT_LOCK = threading.Lock()


def get_salesforce_client(settings: Settings) -> "SalesforceClient":
    """Return the process-wide SalesforceClient, creating on first use."""
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = SalesforceClient(settings)
    return _CLIENT


class SalesforceClient:
    """Wraps simple_salesforce with project-specific queries.

    Auth via username/password/token from Settings. No module-level
    side effects — instance is created when needed.
    """

    def __init__(self, settings: Settings):
        if not settings.sf_username or not settings.sf_password:
            raise ValueError(
                "Salesforce credentials missing — set IDP_SF_USERNAME and "
                "IDP_SF_PASSWORD in environment."
            )
        self._settings = settings
        self._timeout = settings.sf_request_timeout
        self.sf = Salesforce(
            username=settings.sf_username,
            password=settings.sf_password,
            security_token=settings.sf_token or "",
            domain=settings.sf_domain,
        )
        # Default timeout for ALL requests via this session — applies to
        # SOQL queries and the binary download for PDFs. Without this, a
        # hung Salesforce response could block a worker indefinitely.
        self.sf.session.request = self._wrap_with_timeout(self.sf.session.request)
        logger.info(
            "Salesforce client initialized for %s (timeout=%ds)",
            settings.sf_username, self._timeout,
        )

    def _wrap_with_timeout(self, request_fn):
        """Wrap session.request to inject a default timeout when none given."""
        timeout = self._timeout

        def wrapped(method, url, **kwargs):
            kwargs.setdefault("timeout", timeout)
            return request_fn(method, url, **kwargs)

        return wrapped

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def get_case_context(self, case_id: str) -> dict[str, Any]:
        """Lightweight query for the inputs the IDP pipeline needs.

        Returns CertType__c and Funding_Program2__c — both are populated
        on the Case at creation, before any MuleSoft processing.
        """
        case_id_s = _escape_soql(case_id)
        result = self.sf.query(f"""
            SELECT Id, CaseNumber,
                   CertType__c,
                   Funding_Program2__c,
                   Status,
                   IDP_File_Process_Status__c
            FROM Case
            WHERE Id = '{case_id_s}'
        """)
        records = result.get("records", [])
        if not records:
            raise ValueError(f"Case {case_id} not found in Salesforce")
        return records[0]

    def get_case_audit_data(self, case_id: str) -> dict[str, Any]:
        """Full case data for IDP-vs-MuleSoft comparison.

        Pulls Cert Review + Members + Income + Assets so the comparator
        can diff against the IDP extraction.
        """
        case_id_s = _escape_soql(case_id)

        review = self.sf.query(f"""
            SELECT Id, Status__c, Certification_Type__c,
                   Effective_Date__c, Funding_Program__c,
                   Unit_Number__c, Unit_Size__c,
                   Gross_Rent__c, Rent_Limit__c,
                   Income_Limit__c, Set_Aside__c,
                   Total_Income__c, Total_Household_Assets__c,
                   Total_Gross_Household_Income__c,
                   Household_Name__c, Number_Members__c,
                   Passbook_Rate__c,
                   TIC_Tenant_Rent__c, TIC_Utility_Allowance__c,
                   TIC_Gross_Rent__c, TIC_Unit_Set_Aside__c
            FROM Certification_Review__c
            WHERE Case__c = '{case_id_s}'
        """).get("records", [])

        if not review:
            return {
                "review": None,
                "members": [],
                "income": [],
                "assets": [],
            }

        rid = review[0]["Id"]
        rid_s = _escape_soql(rid)

        members = self.sf.query(f"""
            SELECT Id, First_Name__c, Last_Name__c,
                   DOB__c, SSN__c,
                   Disabled__c, Student__c, Head__c,
                   House_Member_Id__c, Full_Name__c,
                   Minor_Member__c,
                   Tenant_Email__c, Tenant_Phone__c
            FROM Household_Member_Cert__c
            WHERE Certification_Review__c = '{rid_s}'
        """).get("records", [])

        income = self.sf.query(f"""
            SELECT Id, Source_Name__c, Income_Type__c,
                   Self_Declared_Amount__c,
                   Self_Declared_Frequency__c,
                   Self_Declared_Annual_Amount__c,
                   Self_Declared_Source__c,
                   Frequency_of_Pay__c,
                   House_Member_Name__c, Household_Member__c,
                   Gross_Member_Income__c,
                   Total_of_VOI_Income__c, Total_of_Paystubs__c,
                   Paystubs_Count_of_Stubs__c,
                   Total_YTD_Income__c,
                   YTD_Start_Date__c, YTD_End_Date__c,
                   Notes__c, Verification_Complete__c
            FROM Income_Worksheet__c
            WHERE Certification_Review__c = '{rid_s}'
        """).get("records", [])

        assets = self.sf.query(f"""
            SELECT Id, Source_Name__c, Account_Type__c,
                   Self_Declared_Balance__c,
                   Self_Declared_Source__c,
                   Cash_Value__c, Income_from_Asset__c,
                   Asset_Cash_Value__c, Asset_Income__c,
                   Interest_Rate__c, Percentage_of_Ownership__c,
                   VOA_Current__c, VOA_Six_Month_Average_Balance__c,
                   VOA_Cash_Value__c, VOA_Net_Value__c,
                   Recent_Bank_Statement_Balance__c,
                   Six_Month_Average__c,
                   Current_Market_Value__c, Net_Value_of_Real_Estate__c,
                   House_Member_Name__c, Household_Member__c,
                   Notes__c, Verification_Complete__c
            FROM Asset_Worksheet__c
            WHERE Certification_Review__c = '{rid_s}'
        """).get("records", [])

        return {
            "review_id": rid,
            "review": review[0],
            "members": members,
            "income": income,
            "assets": assets,
        }

    def download_pdf(self, content_document_id: str) -> bytes:
        """Download the latest PDF version for a ContentDocument."""
        cd_id_s = _escape_soql(content_document_id)
        versions = self.sf.query(f"""
            SELECT Id, Title, FileExtension, VersionData
            FROM ContentVersion
            WHERE ContentDocumentId = '{cd_id_s}'
              AND IsLatest = true
        """).get("records", [])

        if not versions:
            raise ValueError(
                f"No latest ContentVersion for document {content_document_id}"
            )

        version = versions[0]
        if (version.get("FileExtension") or "").lower() != "pdf":
            raise ValueError(
                f"ContentDocument {content_document_id} is not a PDF "
                f"(extension: {version.get('FileExtension')})"
            )

        version_data = version["VersionData"]
        download_url = f"https://{self.sf.sf_instance}{version_data}"
        for attempt in range(3):
            response = self.sf.session.get(
                download_url,
                headers={"Authorization": f"Bearer {self.sf.session_id}"},
            )
            if response.status_code == 200 and len(response.content) > 1000:
                return response.content
            logger.warning(
                "PDF download attempt %d failed: status=%d size=%d",
                attempt + 1, response.status_code, len(response.content),
            )
            time.sleep(2 ** attempt)
        raise RuntimeError(
            f"Failed to download PDF {content_document_id} after 3 attempts"
        )

    def get_pdf_for_case(self, case_id: str) -> tuple[bytes, str]:
        """Find and download the most recent PDF attached to a Case.

        Returns (pdf_bytes, content_document_id).
        Falls back when the webhook doesn't include the document id directly.
        """
        case_id_s = _escape_soql(case_id)
        links = self.sf.query(f"""
            SELECT ContentDocumentId
            FROM ContentDocumentLink
            WHERE LinkedEntityId = '{case_id_s}'
        """).get("records", [])

        for link in links:
            cd_id = link["ContentDocumentId"]
            try:
                pdf_bytes = self.download_pdf(cd_id)
                return pdf_bytes, cd_id
            except ValueError:
                # Not a PDF — try next attachment
                continue

        raise ValueError(f"No PDF attachment found for case {case_id}")
