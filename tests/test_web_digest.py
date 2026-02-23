"""Tests for digest web routes."""


class TestDigestRoutes:
    def test_digest_router_importable(self):
        from pkb.web.routes.digest import router

        assert router is not None

    def test_digest_form_route_exists(self):
        from pkb.web.routes.digest import router

        paths = [r.path for r in router.routes]
        assert "" in paths or "/" in paths or "/digest" in paths

    def test_digest_generate_route_exists(self):
        from pkb.web.routes.digest import router

        methods = set()
        for r in router.routes:
            if hasattr(r, "methods"):
                methods.update(r.methods)
        assert "POST" in methods
