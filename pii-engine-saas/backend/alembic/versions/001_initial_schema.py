"""Initial schema -- all tables for PII Engine SaaS platform.

Revision ID: 001
Revises: None
Create Date: 2026-03-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. tenants
    # ------------------------------------------------------------------
    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False),
        sa.Column("plan", sa.String(50), server_default="free", nullable=False),
        sa.Column("settings", postgresql.JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("max_users", sa.Integer, server_default=sa.text("5"), nullable=False),
        sa.Column("max_api_keys", sa.Integer, server_default=sa.text("3"), nullable=False),
        sa.Column("max_monthly_scans", sa.Integer, server_default=sa.text("1000"), nullable=False),
        sa.Column("is_active", sa.Boolean, server_default=sa.text("true"), nullable=False),
        sa.Column("onboarded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_tenants"),
        sa.UniqueConstraint("slug", name="uq_tenants_slug"),
    )
    op.create_index("ix_tenants_slug", "tenants", ["slug"])
    op.create_index("ix_tenants_plan", "tenants", ["plan"])

    # ------------------------------------------------------------------
    # 2. users
    # ------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("role", sa.String(30), server_default="viewer", nullable=False),
        sa.Column("is_active", sa.Boolean, server_default=sa.text("true"), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("email_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("avatar_url", sa.String(500), nullable=True),
        sa.Column("preferences", postgresql.JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], name="fk_users_tenant_id_tenants", ondelete="CASCADE"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_tenant_id", "users", ["tenant_id"])
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_role", "users", ["role"])

    # ------------------------------------------------------------------
    # 3. user_sessions
    # ------------------------------------------------------------------
    op.create_table(
        "user_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("refresh_token_hash", sa.String(255), nullable=False),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_user_sessions"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_user_sessions_user_id_users", ondelete="CASCADE"),
    )
    op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"])
    op.create_index("ix_user_sessions_expires_at", "user_sessions", ["expires_at"])

    # ------------------------------------------------------------------
    # 4. pii_categories
    # ------------------------------------------------------------------
    op.create_table(
        "pii_categories",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("tier", sa.SmallInteger, nullable=False),
        sa.Column("sort_order", sa.SmallInteger, server_default=sa.text("0"), nullable=False),
        sa.Column("is_system", sa.Boolean, server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_pii_categories"),
        sa.UniqueConstraint("name", name="uq_pii_categories_name"),
    )

    # ------------------------------------------------------------------
    # 5. pii_types
    # ------------------------------------------------------------------
    op.create_table(
        "pii_types",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("category_id", sa.Integer, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("gliner_aliases", postgresql.JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("default_threshold", sa.Numeric(4, 3), server_default=sa.text("0.400"), nullable=False),
        sa.Column("is_system", sa.Boolean, server_default=sa.text("true"), nullable=False),
        sa.Column("is_sensitive", sa.Boolean, server_default=sa.text("false"), nullable=False),
        sa.Column("compliance_tags", postgresql.JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_pii_types"),
        sa.ForeignKeyConstraint(["category_id"], ["pii_categories.id"], name="fk_pii_types_category_id_pii_categories", ondelete="CASCADE"),
        sa.UniqueConstraint("name", name="uq_pii_types_name"),
    )
    op.create_index("ix_pii_types_category_id", "pii_types", ["category_id"])
    op.create_index("ix_pii_types_name", "pii_types", ["name"])

    # ------------------------------------------------------------------
    # 6. tenant_pii_configs
    # ------------------------------------------------------------------
    op.create_table(
        "tenant_pii_configs",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pii_type_id", sa.Integer, nullable=False),
        sa.Column("is_enabled", sa.Boolean, server_default=sa.text("true"), nullable=False),
        sa.Column("custom_threshold", sa.Numeric(4, 3), nullable=True),
        sa.Column("custom_aliases", postgresql.JSONB, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_tenant_pii_configs"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], name="fk_tenant_pii_configs_tenant_id_tenants", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["pii_type_id"], ["pii_types.id"], name="fk_tenant_pii_configs_pii_type_id_pii_types", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], name="fk_tenant_pii_configs_created_by_users", ondelete="SET NULL"),
        sa.UniqueConstraint("tenant_id", "pii_type_id", name="uq_tenant_pii_config"),
    )
    op.create_index("ix_tenant_pii_configs_tenant_id", "tenant_pii_configs", ["tenant_id"])
    op.create_index("ix_tenant_pii_configs_pii_type_id", "tenant_pii_configs", ["pii_type_id"])

    # ------------------------------------------------------------------
    # 7. custom_pii_types
    # ------------------------------------------------------------------
    op.create_table(
        "custom_pii_types",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("category_id", sa.Integer, nullable=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("gliner_aliases", postgresql.JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("default_threshold", sa.Numeric(4, 3), server_default=sa.text("0.400"), nullable=False),
        sa.Column("is_enabled", sa.Boolean, server_default=sa.text("true"), nullable=False),
        sa.Column("compliance_tags", postgresql.JSONB, server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_custom_pii_types"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], name="fk_custom_pii_types_tenant_id_tenants", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["category_id"], ["pii_categories.id"], name="fk_custom_pii_types_category_id_pii_categories", ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], name="fk_custom_pii_types_created_by_users", ondelete="SET NULL"),
        sa.UniqueConstraint("tenant_id", "name", name="uq_custom_pii_type_tenant_name"),
    )
    op.create_index("ix_custom_pii_types_tenant_id", "custom_pii_types", ["tenant_id"])

    # ------------------------------------------------------------------
    # 8. regex_rules
    # ------------------------------------------------------------------
    op.create_table(
        "regex_rules",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("pii_type_name", sa.String(100), nullable=False),
        sa.Column("pattern", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("sort_order", sa.SmallInteger, server_default=sa.text("0"), nullable=False),
        sa.Column("is_system", sa.Boolean, server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_regex_rules"),
    )
    op.create_index("ix_regex_rules_pii_type_name", "regex_rules", ["pii_type_name"])

    # ------------------------------------------------------------------
    # 9. tenant_regex_rules
    # ------------------------------------------------------------------
    op.create_table(
        "tenant_regex_rules",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pii_type_name", sa.String(100), nullable=False),
        sa.Column("pattern", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("is_enabled", sa.Boolean, server_default=sa.text("true"), nullable=False),
        sa.Column("is_system_override", sa.Boolean, server_default=sa.text("false"), nullable=False),
        sa.Column("system_rule_id", sa.Integer, nullable=True),
        sa.Column("sort_order", sa.SmallInteger, server_default=sa.text("0"), nullable=False),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_tenant_regex_rules"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], name="fk_tenant_regex_rules_tenant_id_tenants", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["system_rule_id"], ["regex_rules.id"], name="fk_tenant_regex_rules_system_rule_id_regex_rules", ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], name="fk_tenant_regex_rules_created_by_users", ondelete="SET NULL"),
    )
    op.create_index("ix_tenant_regex_rules_tenant_id", "tenant_regex_rules", ["tenant_id"])
    op.create_index("ix_tenant_regex_rules_pii_type_name", "tenant_regex_rules", ["pii_type_name"])

    # ------------------------------------------------------------------
    # 10. field_patterns
    # ------------------------------------------------------------------
    op.create_table(
        "field_patterns",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("pii_type_name", sa.String(100), nullable=False),
        sa.Column("label_pattern", sa.String(255), nullable=False),
        sa.Column("is_system", sa.Boolean, server_default=sa.text("true"), nullable=False),
        sa.Column("sort_order", sa.SmallInteger, server_default=sa.text("0"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_field_patterns"),
    )
    op.create_index("ix_field_patterns_pii_type_name", "field_patterns", ["pii_type_name"])

    # ------------------------------------------------------------------
    # 11. tenant_field_patterns
    # ------------------------------------------------------------------
    op.create_table(
        "tenant_field_patterns",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pii_type_name", sa.String(100), nullable=False),
        sa.Column("label_pattern", sa.String(255), nullable=False),
        sa.Column("is_enabled", sa.Boolean, server_default=sa.text("true"), nullable=False),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_tenant_field_patterns"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], name="fk_tenant_field_patterns_tenant_id_tenants", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], name="fk_tenant_field_patterns_created_by_users", ondelete="SET NULL"),
        sa.UniqueConstraint("tenant_id", "pii_type_name", "label_pattern", name="uq_tenant_field_pattern"),
    )
    op.create_index("ix_tenant_field_patterns_tenant_id", "tenant_field_patterns", ["tenant_id"])
    op.create_index("ix_tenant_field_patterns_pii_type_name", "tenant_field_patterns", ["pii_type_name"])

    # ------------------------------------------------------------------
    # 12. context_rules
    # ------------------------------------------------------------------
    op.create_table(
        "context_rules",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("pii_type_name", sa.String(100), nullable=False),
        sa.Column("keyword_pattern", sa.Text, nullable=False),
        sa.Column("is_negative", sa.Boolean, server_default=sa.text("false"), nullable=False),
        sa.Column("is_system", sa.Boolean, server_default=sa.text("true"), nullable=False),
        sa.Column("source", sa.String(30), server_default="manual", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_context_rules"),
    )
    op.create_index("ix_context_rules_pii_type_name", "context_rules", ["pii_type_name"])

    # ------------------------------------------------------------------
    # 13. tenant_context_rules
    # ------------------------------------------------------------------
    op.create_table(
        "tenant_context_rules",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pii_type_name", sa.String(100), nullable=False),
        sa.Column("keyword_pattern", sa.Text, nullable=False),
        sa.Column("is_negative", sa.Boolean, server_default=sa.text("false"), nullable=False),
        sa.Column("is_enabled", sa.Boolean, server_default=sa.text("true"), nullable=False),
        sa.Column("source", sa.String(30), server_default="manual", nullable=False),
        sa.Column("promoted_from_id", sa.Integer, nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_tenant_context_rules"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], name="fk_tenant_context_rules_tenant_id_tenants", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], name="fk_tenant_context_rules_created_by_users", ondelete="SET NULL"),
    )
    op.create_index("ix_tenant_context_rules_tenant_id", "tenant_context_rules", ["tenant_id"])
    op.create_index("ix_tenant_context_rules_pii_type_name", "tenant_context_rules", ["pii_type_name"])

    # ------------------------------------------------------------------
    # 14. masking_strategies
    # ------------------------------------------------------------------
    op.create_table(
        "masking_strategies",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("display_name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("implementation_key", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_masking_strategies"),
        sa.UniqueConstraint("name", name="uq_masking_strategies_name"),
    )

    # ------------------------------------------------------------------
    # 15. masking_rules
    # ------------------------------------------------------------------
    op.create_table(
        "masking_rules",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("pii_type_name", sa.String(100), nullable=False),
        sa.Column("strategy_id", sa.Integer, nullable=False),
        sa.Column("is_default", sa.Boolean, server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_masking_rules"),
        sa.ForeignKeyConstraint(["strategy_id"], ["masking_strategies.id"], name="fk_masking_rules_strategy_id_masking_strategies", ondelete="CASCADE"),
    )
    op.create_index("ix_masking_rules_pii_type_name", "masking_rules", ["pii_type_name"])
    op.create_index("ix_masking_rules_strategy_id", "masking_rules", ["strategy_id"])

    # ------------------------------------------------------------------
    # 16. tenant_masking_rules
    # ------------------------------------------------------------------
    op.create_table(
        "tenant_masking_rules",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pii_type_name", sa.String(100), nullable=False),
        sa.Column("strategy_id", sa.Integer, nullable=False),
        sa.Column("custom_params", postgresql.JSONB, nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_tenant_masking_rules"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], name="fk_tenant_masking_rules_tenant_id_tenants", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["strategy_id"], ["masking_strategies.id"], name="fk_tenant_masking_rules_strategy_id_masking_strategies", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], name="fk_tenant_masking_rules_created_by_users", ondelete="SET NULL"),
        sa.UniqueConstraint("tenant_id", "pii_type_name", name="uq_tenant_masking_rule"),
    )
    op.create_index("ix_tenant_masking_rules_tenant_id", "tenant_masking_rules", ["tenant_id"])
    op.create_index("ix_tenant_masking_rules_pii_type_name", "tenant_masking_rules", ["pii_type_name"])

    # ------------------------------------------------------------------
    # 17. api_access_requests
    # ------------------------------------------------------------------
    op.create_table(
        "api_access_requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("requested_by", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("use_case", sa.Text, nullable=False),
        sa.Column("scopes_requested", postgresql.JSONB, server_default=sa.text("'[\"detect\",\"redact\"]'::jsonb"), nullable=False),
        sa.Column("rate_limit_requested", sa.Integer, nullable=True),
        sa.Column("status", sa.String(20), server_default="pending", nullable=False),
        sa.Column("reviewed_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("review_notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_api_access_requests"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], name="fk_api_access_requests_tenant_id_tenants", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["requested_by"], ["users.id"], name="fk_api_access_requests_requested_by_users", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["reviewed_by"], ["users.id"], name="fk_api_access_requests_reviewed_by_users", ondelete="SET NULL"),
    )
    op.create_index("ix_api_access_requests_tenant_id", "api_access_requests", ["tenant_id"])
    op.create_index("ix_api_access_requests_status", "api_access_requests", ["status"])

    # ------------------------------------------------------------------
    # 18. api_keys
    # ------------------------------------------------------------------
    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("access_request_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("key_prefix", sa.String(8), nullable=False),
        sa.Column("key_hash", sa.String(255), nullable=False),
        sa.Column("scopes", postgresql.JSONB, server_default=sa.text("'[\"detect\",\"redact\"]'::jsonb"), nullable=False),
        sa.Column("rate_limit_per_min", sa.Integer, server_default=sa.text("60"), nullable=False),
        sa.Column("is_active", sa.Boolean, server_default=sa.text("true"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_requests", sa.BigInteger, server_default=sa.text("0"), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_api_keys"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], name="fk_api_keys_tenant_id_tenants", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_api_keys_user_id_users", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["access_request_id"], ["api_access_requests.id"], name="fk_api_keys_access_request_id_api_access_requests", ondelete="SET NULL"),
        sa.UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
    )
    op.create_index("ix_api_keys_tenant_id", "api_keys", ["tenant_id"])
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])
    op.create_index("ix_api_keys_key_prefix", "api_keys", ["key_prefix"])
    op.create_index("ix_api_keys_is_active", "api_keys", ["is_active"])

    # ------------------------------------------------------------------
    # 19. api_usage_logs
    # ------------------------------------------------------------------
    op.create_table(
        "api_usage_logs",
        sa.Column("id", sa.BigInteger, autoincrement=True, nullable=False),
        sa.Column("api_key_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("endpoint", sa.String(100), nullable=False),
        sa.Column("method", sa.String(10), nullable=False),
        sa.Column("status_code", sa.SmallInteger, nullable=False),
        sa.Column("response_time_ms", sa.Integer, nullable=True),
        sa.Column("pii_types_found", sa.Integer, server_default=sa.text("0"), nullable=False),
        sa.Column("input_char_count", sa.Integer, nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_api_usage_logs"),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], name="fk_api_usage_logs_api_key_id_api_keys", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], name="fk_api_usage_logs_tenant_id_tenants", ondelete="CASCADE"),
    )
    op.create_index("ix_api_usage_logs_api_key_id", "api_usage_logs", ["api_key_id"])
    op.create_index("ix_api_usage_logs_tenant_id", "api_usage_logs", ["tenant_id"])
    op.create_index("ix_api_usage_logs_created_at", "api_usage_logs", ["created_at"])

    # ------------------------------------------------------------------
    # 20. ml_model_versions
    # ------------------------------------------------------------------
    op.create_table(
        "ml_model_versions",
        sa.Column("id", sa.Integer, autoincrement=True, nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version", sa.Integer, server_default=sa.text("1"), nullable=False),
        sa.Column("model_bucket_path", sa.String(500), nullable=False),
        sa.Column("vocab_bucket_path", sa.String(500), nullable=False),
        sa.Column("num_labels", sa.Integer, nullable=False),
        sa.Column("num_examples", sa.Integer, nullable=False),
        sa.Column("train_size", sa.Integer, nullable=True),
        sa.Column("val_size", sa.Integer, nullable=True),
        sa.Column("val_accuracy", sa.Numeric(5, 4), nullable=True),
        sa.Column("val_weighted_f1", sa.Numeric(5, 4), nullable=True),
        sa.Column("metrics_detail", postgresql.JSONB, nullable=True),
        sa.Column("is_active", sa.Boolean, server_default=sa.text("false"), nullable=False),
        sa.Column("training_trigger", sa.String(30), server_default="auto", nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_ml_model_versions"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], name="fk_ml_model_versions_tenant_id_tenants", ondelete="CASCADE"),
        sa.UniqueConstraint("tenant_id", "version", name="uq_ml_model_tenant_version"),
    )
    op.create_index("ix_ml_model_versions_tenant_id", "ml_model_versions", ["tenant_id"])
    op.create_index("ix_ml_model_versions_is_active", "ml_model_versions", ["is_active"])

    # ------------------------------------------------------------------
    # 21. notifications
    # ------------------------------------------------------------------
    op.create_table(
        "notifications",
        sa.Column("id", sa.BigInteger, autoincrement=True, nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("type", sa.String(50), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("metadata", postgresql.JSONB, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("is_read", sa.Boolean, server_default=sa.text("false"), nullable=False),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_notifications"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], name="fk_notifications_tenant_id_tenants", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_notifications_user_id_users", ondelete="CASCADE"),
    )
    op.create_index("ix_notifications_tenant_id", "notifications", ["tenant_id"])
    op.create_index("ix_notifications_user_id", "notifications", ["user_id"])
    op.create_index("ix_notifications_is_read", "notifications", ["is_read"])
    op.create_index("ix_notifications_type", "notifications", ["type"])


def downgrade() -> None:
    # Drop all tables in reverse dependency order.
    op.drop_table("notifications")
    op.drop_table("ml_model_versions")
    op.drop_table("api_usage_logs")
    op.drop_table("api_keys")
    op.drop_table("api_access_requests")
    op.drop_table("tenant_masking_rules")
    op.drop_table("masking_rules")
    op.drop_table("masking_strategies")
    op.drop_table("tenant_context_rules")
    op.drop_table("context_rules")
    op.drop_table("tenant_field_patterns")
    op.drop_table("field_patterns")
    op.drop_table("tenant_regex_rules")
    op.drop_table("regex_rules")
    op.drop_table("custom_pii_types")
    op.drop_table("tenant_pii_configs")
    op.drop_table("pii_types")
    op.drop_table("pii_categories")
    op.drop_table("user_sessions")
    op.drop_table("users")
    op.drop_table("tenants")
