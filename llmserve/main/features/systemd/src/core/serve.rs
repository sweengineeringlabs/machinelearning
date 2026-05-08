use std::net::SocketAddr;

use anyhow::Result;
use axum::Router;

/// Bind a TCP listener at `addr` and run the axum `router` in the
/// foreground. Returns when the server exits (SIGINT / panic / internal
/// error surfaced by axum).
pub async fn serve_http(addr: SocketAddr, router: Router) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}
