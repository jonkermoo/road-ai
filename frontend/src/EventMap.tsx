import { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { supabase } from "./supabaseClient";
import type { Event } from "./supabaseClient";

// Fix default marker icons in Leaflet (Webpack/Vite issue)
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
});

// Custom marker colors for different event types
const createCustomIcon = (color: string) => {
  return L.divIcon({
    className: "custom-marker",
    html: `<div style="background-color: ${color}; width: 25px; height: 25px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.3);"></div>`,
    iconSize: [25, 25],
    iconAnchor: [12, 12],
  });
};

const eventTypeColors: Record<string, string> = {
  pothole: "#5adc64", // Green
  police: "#3caaff", // Blue
  roadwork: "#ff783c", // Orange
};

export function EventMap() {
  const [events, setEvents] = useState<Event[]>([]);
  const [loading, setLoading] = useState(true);
  const [center, setCenter] = useState<[number, number]>([39.8283, -98.5795]); // Center of USA as default

  const fetchEvents = async () => {
    try {
      const { data, error } = await supabase
        .from("events")
        .select("*")
        .not("lat", "is", null)
        .not("lng", "is", null)
        .order("ts", { ascending: false })
        .limit(100);

      if (error) throw error;

      setEvents(data || []);

      // Center map on most recent event if available
      if (data && data.length > 0 && data[0].lat && data[0].lng) {
        setCenter([data[0].lat, data[0].lng]);
      }
    } catch (error) {
      console.error("Failed to fetch events:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvents();

    // Poll for new events every 5 seconds
    const interval = setInterval(fetchEvents, 5000);

    return () => clearInterval(interval);
  }, []);

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  if (loading) {
    return (
      <div className="bg-gray-800 p-4 rounded-lg">
        <p className="text-center text-gray-400">Loading map...</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 p-4 rounded-lg">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold">Event Map</h2>
        <div className="flex gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div
              className="w-4 h-4 rounded-full"
              style={{ backgroundColor: eventTypeColors.pothole }}
            ></div>
            <span>Pothole</span>
          </div>
          <div className="flex items-center gap-2">
            <div
              className="w-4 h-4 rounded-full"
              style={{ backgroundColor: eventTypeColors.police }}
            ></div>
            <span>Police</span>
          </div>
          <div className="flex items-center gap-2">
            <div
              className="w-4 h-4 rounded-full"
              style={{ backgroundColor: eventTypeColors.roadwork }}
            ></div>
            <span>Roadwork</span>
          </div>
        </div>
      </div>

      {events.length === 0 ? (
        <p className="text-center text-gray-400 py-8">
          No events with GPS data yet
        </p>
      ) : (
        <div className="rounded-lg overflow-hidden" style={{ height: "500px" }}>
          <MapContainer
            center={center}
            zoom={13}
            style={{ height: "100%", width: "100%" }}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {events.map((event) => {
              if (!event.lat || !event.lng) return null;

              const color = eventTypeColors[event.type] || "#999";
              const icon = createCustomIcon(color);

              return (
                <Marker
                  key={event.id}
                  position={[event.lat, event.lng]}
                  icon={icon}
                >
                  <Popup>
                    <div className="text-gray-900">
                      <h3 className="font-bold text-lg capitalize mb-2">
                        {event.type}
                      </h3>
                      <p className="text-sm mb-2">{formatDate(event.ts)}</p>
                      {event.img_url && (
                        <img
                          src={event.img_url}
                          alt={event.type}
                          className="w-full max-w-xs rounded mb-2"
                        />
                      )}
                      <p className="text-xs text-gray-600">
                        Lat: {event.lat.toFixed(6)}, Lng: {event.lng.toFixed(6)}
                      </p>
                    </div>
                  </Popup>
                </Marker>
              );
            })}
          </MapContainer>
        </div>
      )}
    </div>
  );
}
