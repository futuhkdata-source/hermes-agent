from document_processing import (
    DOCUMENT_EXTRACT_SCHEMA,
    check_requirements,
    handle_document_extract,
)


def register(ctx) -> None:
    ctx.register_tool(
        name="document_extract",
        toolset="document_extract",
        schema=DOCUMENT_EXTRACT_SCHEMA,
        handler=handle_document_extract,
        check_fn=check_requirements,
        emoji="📄",
    )
