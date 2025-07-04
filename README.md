# Hansard Spotlight

A tool for enriching Kenya Parliament proceedings published in [Hansard][1]
with context similar to Wikipedia. Inspired by [DBPedia spotlight][2]


## Installation

This project uses [uv][3] for dependency management. Clone the repo and run `uv
sync`

## Running

Run `src/spotlight/main.py` with a Hansard PDF file as an argument to list the
headers in the document. For example

```sh
uv run -- src/spotlight/main.py "Hansard Report - Wednesday, 30th April 2025 (A).pdf"
```

Which produces the following output

```txt
THE HANSARD
Wednesday, 30th April 2025
The House met at 9.30 a.m.
PRAYERS
QUORUM
PAPERS
NOTICE OF MOTION
ADOPTION OF THIRD REPORT ON AUDIT OF EMPLOYMENT DIVERSITY
QUESTIONS AND STATEMENTS
PROCEDURAL MOTION
EXEMPTION OF BUSINESS FROM THE PROVISIONS OF STANDING ORDER 40(3)
MOTION
BILL
Third Reading
THE SOCIAL PROTECTION BILL (National Assembly Bill No.12 of 2025)
BILL
First Reading
THE INDUSTRIAL TRAINING BILL
QUESTIONS AND STATEMENTS
REQUESTS FOR STATEMENTS
DEMISE OF MS WANJIRU NG’ANG’A MUTONGA
THE STATE OF INFRASTRUCTURE OF KENYA POWER IN NYATIKE CONSTITUENCY
STATUS OF CONSTRUCTION OF LUNGALUNGA SUB-COUNTY HEADQUARTERS
IMPACT OF WITHDRAWAL OF USAID FUNDING
WELFARE OF KENYANS WORKING IN MIDDLE EAST
UNYAKUZI WA ARDHI YA MSIKITI WA KONGO KATIKA ENEOBUNGE LA MSAMBWENI
REQUESTS FOR STATEMENTS
HUMAN-WILDLIFE CONFLICT IN CHEPALUNGU CONSTITUENCY
STATEMENT
RESPONSES TO STATEMENTS
STATUS OF MAVOKO WATER SUPPLY PROJECT
STATUS OF UPGRADING NEW KCC MILK PROCESSOR
SPECIAL MOTION
APPROVAL OF NOMINEES FOR APPOINTMENT AS MEMBERS OF CBK BOARD
ADJOURNMENT
The House rose at 1:00 p.m.
Published by Clerk of the National Assembly
Parliament Buildings Nairobi
```


## Tasks

- [ ] Use detected headers to split PDF file into segments.
- [ ] Create database and load election, parliament, and geographical datasets.
- [ ] Add Hansard sections into database.
- [ ] Create enrichment pipeline.
- [ ] Combine all functionality into web application.


[1]:http://parliament.go.ke/the-national-assembly/house-business/hansard
[2]:https://www.dbpedia-spotlight.org/
[3]:https://docs.astral.sh/uv/
