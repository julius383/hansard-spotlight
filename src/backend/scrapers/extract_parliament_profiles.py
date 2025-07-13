import asyncio
import re
import sys
import json

from functools import partial
from typing import Optional

import aiohttp

from bs4 import BeautifulSoup
from toolz import pipe, partition
from tqdm.asyncio import tqdm
from loguru import logger

#  from icecream import ic
import pysnooper

fmt = "{time} - {name} - {level} - {message}"
logger.add("app.log", level="INFO" ,format=fmt)


MAX_CONNECTIONS_PER_HOST = 5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/140.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,",
    "Accept-Encoding": "gzip, deflate, br",
    "Pragma": "no-cache",
}


def get_website_list(filename: str) -> set[str]:
    websites = set()
    with open(filename, "r", encoding="utf-8") as infile:
        for line in infile:
            websites.add(line)
    return websites


async def fetch_page(session: aiohttp.ClientSession, url: str):
    try:
        async with session.get(url, ssl=False) as response:
            response_url = str(response.url)
            if response.status == 200:
                text = await response.read()
                return text, response_url
            logger.error(f"Network Error - {url} failed with status {response.status}")
            return None, response_url
    except Exception:
        logger.exception("Error trying to fetch: {url} ")
        return None, None
    return None, None


def find_name(soup: BeautifulSoup) -> Optional[dict[str, str]]:
    maybe_name = soup.select("article h1")
    name = pipe(maybe_name[0].text, str.strip)
    return {"name": name}


def find_photo(soup: BeautifulSoup) -> Optional[dict[str, str]]:
    img_ele = soup.select("article img")
    src = None
    if img_ele:
        src = img_ele[0].attrs.get("src")
        if src.startswith('/'):
            src = "http://www.parliament.go.ke" + src
    return {"profile_photo": src}


def find_seat(soup: BeautifulSoup) -> Optional[dict[str, str]]:
    seat_type_ele = soup.select("article h3:nth-child(1)")
    seat = None
    seat_type = None
    if seat_type_ele:
        seat_ele = soup.select("article h3:nth-child(1) + ul > li")
        if seat_ele:
            seat_type = pipe(seat_type_ele[0].text, str.lower, str.strip)
            seat = pipe(seat_ele[0].text, str.strip)
    return {"type": seat_type, "seat": seat}


def find_party(soup: BeautifulSoup) -> Optional[dict[str, str]]:
    party_ele = soup.select("article h3:nth-child(3) + ul > li")
    party = None
    if party_ele:
        party = pipe(party_ele[0].text, str.upper, str.strip)
    return {"party_short": party}


def normalize_key(string):
    return pipe(string, str.lower, str.strip)


def clean_entry(string):
    return pipe(
        string,
        str.strip,
        partial(re.sub, r"\s+", " "),
        partial(re.sub, r"[^a-zA-Z0-9 ]", ""),
    )


# @pysnooper.snoop()
def process_rows(rows, header):
    hs = list(map(normalize_key, header))
    data_rows = rows
    r = []
    patterns = [
        re.compile(r"^\d{4}"),
        re.compile(r"^\d{4}"),
        re.compile(r"^[a-zA-Z]+.+"),
        re.compile(r"^[a-zA-Z]+.+"),
    ]
    while data_rows:
        row = list(map(clean_entry, data_rows.pop(0)))
        if re.match(r"^\d{4}\s+\d{4}$", row[0]):
            start_years = re.split(r"\s", row[0])
            end_years = re.split(r"\s", row[1])
            schools = re.split(r"\s(?=[A-Z])", row[2])
            quals = re.split(r"\s(?=[A-Z])", row[3])
            try:
                data_rows.insert(
                    0, [start_years[0], end_years[0], schools[0], quals[0]]
                )
                data_rows.insert(
                    0, [start_years[1], end_years[1], schools[1], quals[1]]
                )
                continue
            except IndexError:
                logger.error("failed to process {row}")
                continue
        # if all(re.match(r"\d{4}", x) for x in row):
        #     from_, to = row[:2], row[2:]
        #     row2 = data_rows.pop(0)
        #     name, qual = row2[:2], row2[2:]
        #     data_rows.insert(0, [from_[0], to[0], name[0], qual[0]])
        #     data_rows.insert(0, [from_[1], to[1], name[1], qual[1]])
        #     continue
        if all(re.match(a, b) for (a, b) in zip(patterns, row)):
            cleaned_row = []
            for c in row:
                if m := re.match(r"\d{4}", c):
                    cleaned_row.append(int(m.group()))
                    continue
                cleaned_row.append(c)
            r.append(dict(zip(hs, cleaned_row)))
    return r


def find_education_history(soup: BeautifulSoup) -> Optional[dict[str, str]]:
    table_eles = soup.select(
        'article div[class*="education-background"] table tr td'
    )
    rows = None
    if table_eles:
        elements = list(partition(4, map(lambda x: x.text, table_eles)))
        elements.pop(0)  # remove header row
        rows = process_rows(
            elements,
            ["from", "to", "educational institution", "qualification"],
        )
    return {"education_history": rows}


def find_employment_history(soup: BeautifulSoup) -> Optional[dict[str, str]]:
    table_eles = soup.select(
        'article div[class*="employment-history"] table tr td'
    )
    rows = None
    if table_eles:
        elements = list(partition(4, map(lambda x: x.text, table_eles)))
        elements.pop(0)  # remove header row
        rows = process_rows(
            elements,
            ["from", "to", "employer", "position"],
        )
    return {"employment_history": rows}


def extract_profile_info(soup: BeautifulSoup) -> str:
    d = {}
    for f in [
        find_name,
        find_photo,
        find_seat,
        find_party,
        find_education_history,
        find_employment_history,
    ]:
        d |= f(soup)
    return d


async def main():
    loop = asyncio.get_running_loop()
    urls = await loop.run_in_executor(
        None, partial(get_website_list, sys.argv[1])
    )
    connector = aiohttp.TCPConnector(limit_per_host=MAX_CONNECTIONS_PER_HOST)
    tasks = []
    async with aiohttp.ClientSession(
        headers=HEADERS,
        connector=connector,
    ) as session:
        for url in urls:
            tasks.append(fetch_page(session, url))
        results = await tqdm.gather(*tasks)
        for result in results:
            text, url = result
            if text:
                print(url)
                with open(sys.argv[2], "a", encoding="utf-8") as ofile:
                    soup = BeautifulSoup(text, "html.parser")
                    out = extract_profile_info(soup)
                    out |= {"profile_url": url}
                    ofile.write(json.dumps(out) + "\n")
    return


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        asyncio.run(main())
