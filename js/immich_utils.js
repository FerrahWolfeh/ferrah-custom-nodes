import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Extension for the immich_upload node that fetches albums dynamically.
 */
app.registerExtension({
    name: "Ferrah.ImmichUpload",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "immich_upload") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const node = this;
                const albumIdWidget = node.widgets.find(w => w.name === "album_id");
                
                if (!albumIdWidget) return r;

                // Function to fetch albums and update the widget to a combo box
                const fetchAlbums = async () => {
                    try {
                        // The backend uses config.json for credentials
                        const response = await api.fetchApi("/immich/get_albums", {
                            method: "GET"
                        });
                        
                        if (response.status !== 200) {
                            console.warn("ImmichUpload: Could not fetch albums. Check your config.json.");
                            return;
                        }
                        
                        const albums = await response.json();
                        if (!albums || albums.error) {
                            console.error("Immich API Error:", albums?.error);
                            return;
                        }

                        // Prepare options in "Name (ID)" format
                        const options = albums.map(a => `${a.name} (${a.id})`);
                        
                        // Convert widget from STRING to COMBO dynamically
                        albumIdWidget.type = "combo";
                        if (!albumIdWidget.options) albumIdWidget.options = {};
                        albumIdWidget.options.values = options;
                        
                        // Try to keep the current value if it's valid
                        if (albumIdWidget.value && !options.includes(albumIdWidget.value)) {
                            const match = options.find(o => o.endsWith(`(${albumIdWidget.value})`));
                            if (match) {
                                albumIdWidget.value = match;
                            }
                        }
                        
                        node.setDirtyCanvas(true, true);
                    } catch (e) {
                        console.error("Error fetching Immich albums:", e);
                    }
                };

                // Initial fetch
                setTimeout(fetchAlbums, 1000);
                
                // Add Refresh option to context menu
                const onGetExtraMenuOptions = node.getExtraMenuOptions;
                node.getExtraMenuOptions = function(_, options) {
                    if (onGetExtraMenuOptions) onGetExtraMenuOptions.apply(this, arguments);
                    options.push({
                        content: "↻ Refresh Immich Albums",
                        callback: () => fetchAlbums()
                    });
                };

                return r;
            };
        }
    }
});
