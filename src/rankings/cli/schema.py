from __future__ import annotations

"""
Introspect and print the database schema for rankings tables.

Usage examples:
  poetry run rankings_db_schema
  poetry run rankings_db_schema --schema comp_rankings --format json
  poetry run rankings_db_schema --db-url postgresql://user:pass@host/db?sslmode=require

Reads connection info from env the same way as other rankings CLI tools.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from rankings.sql import create_engine as rankings_create_engine
from rankings.sql.constants import SCHEMA as DEFAULT_SCHEMA


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    udt_name: Optional[str]
    is_nullable: bool
    default: Optional[str]
    character_max_length: Optional[int]
    numeric_precision: Optional[int]
    numeric_scale: Optional[int]


@dataclass
class IndexInfo:
    name: str
    definition: str


@dataclass
class ConstraintInfo:
    name: str
    type: str  # PRIMARY KEY, UNIQUE, FOREIGN KEY, CHECK
    columns: List[str]
    details: Optional[str]  # For FKs (ref table/cols) or CHECK clause


@dataclass
class TableSchema:
    table: str
    columns: List[ColumnInfo]
    indexes: List[IndexInfo]
    constraints: List[ConstraintInfo]


def _fetch_tables(engine, schema: str) -> List[str]:
    sql = text(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
    )
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(sql, {"schema": schema}).fetchall()]


def _fetch_columns(engine, schema: str, table: str) -> List[ColumnInfo]:
    sql = text(
        """
        SELECT column_name,
               data_type,
               udt_name,
               is_nullable,
               column_default,
               character_maximum_length,
               numeric_precision,
               numeric_scale
          FROM information_schema.columns
         WHERE table_schema = :schema AND table_name = :table
         ORDER BY ordinal_position
        """
    )
    rows = []
    with engine.connect() as conn:
        rows = conn.execute(sql, {"schema": schema, "table": table}).fetchall()
    cols: List[ColumnInfo] = []
    for (
        name,
        data_type,
        udt_name,
        is_nullable,
        default,
        char_len,
        num_prec,
        num_scale,
    ) in rows:
        cols.append(
            ColumnInfo(
                name=name,
                data_type=str(data_type) if data_type is not None else "",
                udt_name=str(udt_name) if udt_name is not None else None,
                is_nullable=(str(is_nullable).lower() == "yes"),
                default=(str(default) if default is not None else None),
                character_max_length=(
                    int(char_len) if char_len is not None else None
                ),
                numeric_precision=(
                    int(num_prec) if num_prec is not None else None
                ),
                numeric_scale=(
                    int(num_scale) if num_scale is not None else None
                ),
            )
        )
    return cols


def _fetch_indexes(engine, schema: str, table: str) -> List[IndexInfo]:
    sql = text(
        """
        SELECT indexname, indexdef
          FROM pg_indexes
         WHERE schemaname = :schema AND tablename = :table
         ORDER BY indexname
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"schema": schema, "table": table}).fetchall()
    return [IndexInfo(name=r[0], definition=r[1]) for r in rows]


def _fetch_constraints(engine, schema: str, table: str) -> List[ConstraintInfo]:
    cons: List[ConstraintInfo] = []
    with engine.connect() as conn:
        # Primary/Unique constraints and their columns
        pk_uniq = conn.execute(
            text(
                """
                SELECT tc.constraint_name,
                       tc.constraint_type,
                       array_agg(kcu.column_name ORDER BY kcu.ordinal_position) AS columns
                  FROM information_schema.table_constraints tc
                  JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                   AND tc.table_schema = kcu.table_schema
                   AND tc.table_name = kcu.table_name
                 WHERE tc.table_schema = :schema
                   AND tc.table_name = :table
                   AND tc.constraint_type IN ('PRIMARY KEY','UNIQUE')
                 GROUP BY tc.constraint_name, tc.constraint_type
                """
            ),
            {"schema": schema, "table": table},
        ).fetchall()
        for name, ctype, cols in pk_uniq:
            cons.append(
                ConstraintInfo(
                    name=name,
                    type=str(ctype),
                    columns=[str(c) for c in (cols or [])],
                    details=None,
                )
            )

        # Foreign keys with referenced table/columns
        fks = conn.execute(
            text(
                """
                SELECT tc.constraint_name,
                       array_agg(kcu.column_name ORDER BY kcu.ordinal_position) AS columns,
                       ccu.table_name AS referenced_table,
                       array_agg(ccu.column_name ORDER BY kcu.ordinal_position) AS referenced_columns
                  FROM information_schema.table_constraints AS tc
                  JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                   AND tc.table_schema = kcu.table_schema
                  JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                   AND ccu.table_schema = tc.table_schema
                 WHERE tc.constraint_type = 'FOREIGN KEY'
                   AND tc.table_schema = :schema
                   AND tc.table_name = :table
                 GROUP BY tc.constraint_name, ccu.table_name
                """
            ),
            {"schema": schema, "table": table},
        ).fetchall()
        for name, cols, ref_table, ref_cols in fks:
            details = f"references {ref_table}({', '.join(ref_cols or [])})"
            cons.append(
                ConstraintInfo(
                    name=name,
                    type="FOREIGN KEY",
                    columns=[str(c) for c in (cols or [])],
                    details=details,
                )
            )

        # Check constraints
        checks = conn.execute(
            text(
                """
                SELECT tc.constraint_name, cc.check_clause
                  FROM information_schema.table_constraints tc
                  JOIN information_schema.check_constraints cc
                    ON tc.constraint_name = cc.constraint_name
                 WHERE tc.table_schema = :schema
                   AND tc.table_name = :table
                   AND tc.constraint_type = 'CHECK'
                 ORDER BY tc.constraint_name
                """
            ),
            {"schema": schema, "table": table},
        ).fetchall()
        for name, clause in checks:
            cons.append(
                ConstraintInfo(
                    name=name,
                    type="CHECK",
                    columns=[],
                    details=str(clause) if clause is not None else None,
                )
            )
    return cons


def _gather_schema(
    engine, schema: str, tables: Optional[List[str]] = None
) -> List[TableSchema]:
    all_tables = tables or _fetch_tables(engine, schema)
    result: List[TableSchema] = []
    for t in all_tables:
        cols = _fetch_columns(engine, schema, t)
        idxs = _fetch_indexes(engine, schema, t)
        cons = _fetch_constraints(engine, schema, t)
        result.append(
            TableSchema(table=t, columns=cols, indexes=idxs, constraints=cons)
        )
    return result


def _print_text(schemas: List[TableSchema], schema: str) -> None:
    print(f"Schema: {schema}")
    for ts in schemas:
        print(f"\n== {ts.table} ==")
        print("Columns:")
        for c in ts.columns:
            dtype = c.udt_name or c.data_type
            null = "NULL" if c.is_nullable else "NOT NULL"
            extras: List[str] = []
            if c.character_max_length is not None:
                extras.append(f"len={c.character_max_length}")
            if c.numeric_precision is not None:
                extras.append(f"prec={c.numeric_precision}")
            if c.numeric_scale is not None:
                extras.append(f"scale={c.numeric_scale}")
            extra_str = f" ({', '.join(extras)})" if extras else ""
            default = f" default {c.default}" if c.default else ""
            print(f"  - {c.name}: {dtype} {null}{extra_str}{default}")
        if ts.indexes:
            print("Indexes:")
            for i in ts.indexes:
                print(f"  - {i.name}: {i.definition}")
        if ts.constraints:
            print("Constraints:")
            for con in ts.constraints:
                cols = f" ({', '.join(con.columns)})" if con.columns else ""
                details = f" [{con.details}]" if con.details else ""
                print(f"  - {con.type}: {con.name}{cols}{details}")


def _print_json(schemas: List[TableSchema], schema: str) -> None:
    obj: Dict[str, Any] = {
        "schema": schema,
        "tables": [
            {
                "table": ts.table,
                "columns": [asdict(c) for c in ts.columns],
                "indexes": [asdict(i) for i in ts.indexes],
                "constraints": [asdict(c) for c in ts.constraints],
            }
            for ts in schemas
        ],
    }
    print(json.dumps(obj, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect rankings DB schema")
    parser.add_argument("--db-url", type=str, default=None, help="Database URL")
    parser.add_argument(
        "--schema",
        type=str,
        default=os.getenv("RANKINGS_DB_SCHEMA", DEFAULT_SCHEMA),
        help="Database schema name",
    )
    parser.add_argument(
        "--tables",
        type=str,
        nargs="*",
        default=None,
        help="Specific tables to inspect (default: all in schema)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    args = parser.parse_args(argv)

    engine = rankings_create_engine(args.db_url)
    schemas = _gather_schema(engine, args.schema, args.tables)
    if args.format == "json":
        _print_json(schemas, args.schema)
    else:
        _print_text(schemas, args.schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
