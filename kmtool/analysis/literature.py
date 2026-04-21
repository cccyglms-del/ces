import re
import xml.etree.ElementTree as ET

import requests
from requests import exceptions as request_exceptions

from kmtool.models import StudyCandidate


PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
REQUEST_HEADERS = {
    "User-Agent": "KM-Indirect-Comparison-Lab/0.1 (+local research app)",
}


def _safe_request(url, params, timeout=25):
    try:
        return _perform_request(url, params=params, timeout=timeout, trust_env=True)
    except Exception as exc:
        if not _looks_like_proxy_error(exc):
            raise
        try:
            return _perform_request(url, params=params, timeout=timeout, trust_env=False)
        except Exception as direct_exc:
            raise RuntimeError(
                "Request failed through the configured system proxy and also failed on direct connection. "
                "Check whether the proxy is running or disable the system proxy.\n"
                "Proxy-layer error: {0}\n"
                "Direct-connection error: {1}".format(exc, direct_exc)
            ) from direct_exc


def _perform_request(url, params, timeout, trust_env):
    with requests.Session() as session:
        session.trust_env = trust_env
        session.headers.update(REQUEST_HEADERS)
        response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response


def _looks_like_proxy_error(exc):
    if isinstance(exc, request_exceptions.ProxyError):
        return True
    if isinstance(exc, request_exceptions.ConnectionError):
        return "proxy" in str(exc).lower()
    return "proxy" in str(exc).lower()


def build_comparison_query(treatment_left, treatment_right, endpoint="", population=""):
    terms = ['"{0}"'.format(treatment_left), '"{0}"'.format(treatment_right), "(trial OR comparative OR randomized)"]
    if endpoint:
        terms.append('"{0}"'.format(endpoint))
    if population:
        terms.append('"{0}"'.format(population))
    return " AND ".join(terms)


def extract_reported_hr(text):
    if not text:
        return None, None, None
    compact = " ".join(text.split())
    hr_match = re.search(r"(?:hazard ratio|hr)\s*(?:=|:)?\s*([0-9]+(?:\.[0-9]+)?)", compact, re.IGNORECASE)
    ci_match = re.search(
        r"95%\s*ci[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*(?:-|to|through|until|[^0-9]{1,6})\s*([0-9]+(?:\.[0-9]+)?)",
        compact,
        re.IGNORECASE,
    )
    if not hr_match:
        return None, None, None
    hr = float(hr_match.group(1))
    if not ci_match:
        return hr, None, None
    return hr, float(ci_match.group(1)), float(ci_match.group(2))


def _pubmed_search_ids(query, retmax=12):
    response = _safe_request(
        PUBMED_SEARCH_URL,
        {"db": "pubmed", "term": query, "retmode": "json", "retmax": retmax},
    )
    payload = response.json()
    return payload.get("esearchresult", {}).get("idlist", [])


def _pubmed_fetch_records(pmids):
    if not pmids:
        return []
    response = _safe_request(
        PUBMED_FETCH_URL,
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
    )
    root = ET.fromstring(response.text)
    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")
        title = "".join(article.find(".//ArticleTitle").itertext()) if article.find(".//ArticleTitle") is not None else ""
        abstract_nodes = article.findall(".//Abstract/AbstractText")
        abstract = " ".join("".join(node.itertext()) for node in abstract_nodes)
        journal = article.findtext(".//Journal/Title", default="")
        year_text = article.findtext(".//PubDate/Year", default="0")
        doi = ""
        for identifier in article.findall(".//ArticleId"):
            if identifier.attrib.get("IdType") == "doi":
                doi = identifier.text or ""
                break
        try:
            year = int(year_text)
        except ValueError:
            year = None
        records.append(
            {
                "source": "PubMed",
                "study_id": "pmid:{0}".format(pmid),
                "pmid": pmid,
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
            }
        )
    return records


def _europe_pmc_search(query, page_size=12):
    response = _safe_request(
        EUROPE_PMC_SEARCH_URL,
        {"query": query, "format": "json", "pageSize": page_size},
    )
    payload = response.json()
    results = payload.get("resultList", {}).get("result", [])
    records = []
    for item in results:
        records.append(
            {
                "source": "Europe PMC",
                "study_id": "epmc:{0}".format(item.get("id", "")),
                "pmid": item.get("pmid", ""),
                "doi": item.get("doi", ""),
                "title": item.get("title", ""),
                "abstract": item.get("abstractText", ""),
                "journal": item.get("journalTitle", ""),
                "year": int(item.get("pubYear")) if item.get("pubYear", "").isdigit() else None,
                "open_access_url": item.get("pmcid", ""),
            }
        )
    return records


def _comparison_type_for_record(text, treatment_left, treatment_right):
    lower = text.lower()
    left = treatment_left.lower()
    right = treatment_right.lower()
    if left in lower and right in lower:
        return "{0} vs {1}".format(treatment_left, treatment_right)
    return ""


def _candidate_from_record(record, treatment_left, treatment_right, endpoint, population):
    text = "{0} {1}".format(record.get("title", ""), record.get("abstract", ""))
    comparison_type = _comparison_type_for_record(text, treatment_left, treatment_right)
    hr, ci_low, ci_high = extract_reported_hr(text)
    warnings = []
    if hr is None:
        warnings.append("Reported HR not found in title/abstract; a local figure/PDF may be needed.")
    return StudyCandidate(
        source=record.get("source", ""),
        study_id=record.get("study_id", ""),
        title=record.get("title", ""),
        abstract=record.get("abstract", ""),
        year=record.get("year"),
        journal=record.get("journal", ""),
        pmid=record.get("pmid", ""),
        doi=record.get("doi", ""),
        comparison_type=comparison_type,
        treatments=[treatment_left, treatment_right],
        endpoint_text=endpoint,
        population_text=population,
        open_access_url=record.get("open_access_url", ""),
        reported_hr=hr,
        ci_low=ci_low,
        ci_high=ci_high,
        warnings=warnings,
    )


def search_comparison_candidates(treatment_left, treatment_right, endpoint="", population="", retmax=12):
    query = build_comparison_query(treatment_left, treatment_right, endpoint=endpoint, population=population)
    pmids = _pubmed_search_ids(query, retmax=retmax)
    pubmed_records = _pubmed_fetch_records(pmids)
    europe_records = _europe_pmc_search(query, page_size=retmax)

    merged = {}
    for record in pubmed_records + europe_records:
        key = record.get("pmid") or record.get("doi") or record.get("title", "").lower()
        if key not in merged:
            merged[key] = record
        else:
            merged[key].update({field: value for field, value in record.items() if value})

    candidates = [
        _candidate_from_record(record, treatment_left, treatment_right, endpoint, population)
        for record in merged.values()
    ]
    candidates = [candidate for candidate in candidates if candidate.comparison_type]
    candidates.sort(key=lambda item: (item.reported_hr is None, -(item.year or 0), item.source))
    return candidates
