from PyInstaller.utils.hooks import collect_data_files


datas = collect_data_files(
    "pywebio",
    includes=[
        "platform/tpl/index.html",
        "html/css/app.css",
        "html/css/markdown.min.css",
        "html/css/codemirror.min.css",
        "html/css/toastify.min.css",
        "html/css/bootstrap-select.min.css",
        # The packaged app enters through a wrapper function, so PyWebIO falls
        # back to the default theme during initial page rendering.
        "html/css/bs-theme/default.min.css",
        "html/css/bs-theme/dark.min.css",
        "html/js/mustache.min.js",
        "html/js/codemirror.min.js",
        "html/js/prism.min.js",
        "html/js/FileSaver.min.js",
        "html/js/jquery.min.js",
        "html/js/popper.min.js",
        "html/js/bootstrap.min.js",
        "html/js/toastify.min.js",
        "html/js/bs-custom-file-input.min.js",
        "html/js/purify.min.js",
        "html/js/bootstrap-select.min.js",
        "html/js/pywebio.min.js",
        "html/js/require.min.js",
        "html/codemirror/*.js",
    ],
)
