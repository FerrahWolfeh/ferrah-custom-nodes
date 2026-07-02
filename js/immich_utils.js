import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Cache for fetched albums to share across multiple nodes
let albumsCache = null;
let albumsPromise = null;

/**
 * Fetch albums from the Immich API with request sharing and caching.
 */
async function getImmichAlbums(forceRefresh = false) {
    if (!forceRefresh && albumsCache) {
        return albumsCache;
    }
    if (albumsPromise && !forceRefresh) {
        return albumsPromise;
    }

    albumsPromise = (async () => {
        try {
            console.log("FerrahNodes: Fetching Immich albums...");
            const response = await api.fetchApi("/immich/get_albums", {
                method: "GET"
            });

            if (response.status !== 200) {
                console.warn("ImmichUpload: Could not fetch albums. Status:", response.status);
                return null;
            }

            const albums = await response.json();
            if (!albums || albums.error) {
                console.error("Immich API Error:", albums?.error);
                return null;
            }

            albumsCache = albums;
            return albums;
        } catch (e) {
            console.error("Error fetching Immich albums:", e);
            return null;
        } finally {
            albumsPromise = null;
        }
    })();

    return albumsPromise;
}

/**
 * Refresh the album options and selections on all Immich Upload nodes in the graph.
 */
async function refreshAllNodes(forceRefresh = false) {
    const albums = await getImmichAlbums(forceRefresh);
    if (!albums) return;

    // Add "(none)" to the options list
    const options = ["(none)", ...albums.map(a => `${a.name} (${a.id})`)];

    // Find all immich_upload and immich_album nodes in the graph
    if (!app.graph || typeof app.graph.findNodesByType !== "function") {
        return;
    }

    const uploadNodes = app.graph.findNodesByType("immich_upload") || [];
    const albumNodes = app.graph.findNodesByType("immich_album") || [];
    const nodes = [...uploadNodes, ...albumNodes];

    for (const node of nodes) {
        const albumIdWidget = node.widgets.find(w => w.name === "album_id");
        if (!albumIdWidget) continue;

        let finalOptions = [...options];
        const currentValue = albumIdWidget.value;
        let resolvedValue = currentValue;

        if (currentValue && currentValue !== "(none/loading)" && currentValue !== "(none)") {
            let hasMatch = finalOptions.includes(currentValue);
            if (!hasMatch) {
                // If it's a UUID, try to match it with Name (UUID) format
                const uuidMatch = finalOptions.find(o => o.endsWith(`(${currentValue})`));
                if (uuidMatch) {
                    resolvedValue = uuidMatch;
                    hasMatch = true;
                }
            }
            // If still no match (e.g. server config changed, or album deleted),
            // prepend it to finalOptions so the saved state is not lost in the UI
            if (!hasMatch) {
                finalOptions.unshift(currentValue);
            }
        } else {
            // Default to "(none)" if no valid selection, currently loading, or set to "(none)"
            if (finalOptions.includes("(none)")) {
                resolvedValue = "(none)";
            } else if (finalOptions.length > 0) {
                resolvedValue = finalOptions[0];
            }
        }

        // Clean up the custom getter/setter on value defined during node creation.
        // This restores default value assignment and display behavior.
        try {
            delete albumIdWidget.value;
        } catch (e) {
            console.warn("Could not delete custom getter/setter on albumIdWidget value:", e);
        }

        albumIdWidget.value = resolvedValue;
        albumIdWidget.options.values = finalOptions;
        node.setDirtyCanvas(true, true);
    }
}

/**
 * Extension for the immich_upload and immich_album nodes that fetches albums dynamically.
 */
app.registerExtension({
    name: "Ferrah.ImmichUpload",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "immich_upload" || nodeData.name === "immich_album") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const node = this;
                const albumIdWidget = node.widgets.find(w => w.name === "album_id");

                if (!albumIdWidget) return r;

                // Intercept the widget's value using a getter/setter.
                // This ensures that when ComfyUI deserializes the workflow and assigns the value
                // (before options are fetched), the value is preserved in options.values.
                let widgetValue = albumIdWidget.value;
                Object.defineProperty(albumIdWidget, "value", {
                    set(v) {
                        widgetValue = v;
                        if (v && v !== "(none/loading)" && v !== "(none)") {
                            if (!albumIdWidget.options.values.includes(v)) {
                                albumIdWidget.options.values = [v];
                            }
                        }
                    },
                    get() {
                        return widgetValue;
                    },
                    configurable: true,
                    enumerable: true
                });

                // Fetch and update all nodes on load
                setTimeout(() => {
                    refreshAllNodes(false).catch(err => console.error("Error refreshing Immich nodes:", err));
                }, 1000);

                // Add Refresh option to context menu
                const onGetExtraMenuOptions = node.getExtraMenuOptions;
                node.getExtraMenuOptions = function (_, options) {
                    if (onGetExtraMenuOptions) onGetExtraMenuOptions.apply(this, arguments);
                    options.push({
                        content: "↻ Refresh Immich Albums",
                        callback: () => {
                            refreshAllNodes(true).catch(err => console.error("Error refreshing Immich nodes:", err));
                        }
                    });
                };

                return r;
            };
        }
    }
});

