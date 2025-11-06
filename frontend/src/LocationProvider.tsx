import { useEffect, useState, useRef } from "react";

interface LocationProviderProps {
  deviceId: string | null;
  isActive: boolean;
}

export function LocationProvider({
  deviceId,
  isActive,
}: LocationProviderProps) {
  const [location, setLocation] = useState<{ lat: number; lng: number } | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);
  const [permissionStatus, setPermissionStatus] = useState<
    "prompt" | "granted" | "denied"
  >("prompt");
  const watchIdRef = useRef<number | null>(null);
  const updateIntervalRef = useRef<number | null>(null);

  const sendLocationToBackend = async (lat: number, lng: number) => {
    if (!deviceId || !isActive) return;

    try {
      await fetch(`/update-location?device_id=${deviceId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat, lng }),
      });
    } catch (error) {
      console.error("Failed to send location:", error);
    }
  };

  const handleLocationUpdate = (position: GeolocationPosition) => {
    const lat = position.coords.latitude;
    const lng = position.coords.longitude;
    setLocation({ lat, lng });
    setError(null);

    // Send to backend
    sendLocationToBackend(lat, lng);
  };

  const handleLocationError = (err: GeolocationPositionError) => {
    console.error("Location error:", err);
    setError(err.message);
  };

  const startTracking = () => {
    if (!navigator.geolocation) {
      setError("Geolocation is not supported by your browser");
      return;
    }

    // Request permission and start watching
    navigator.permissions
      .query({ name: "geolocation" })
      .then((result) => {
        setPermissionStatus(result.state as "prompt" | "granted" | "denied");

        if (result.state === "granted" || result.state === "prompt") {
          // Watch position for real-time updates
          watchIdRef.current = navigator.geolocation.watchPosition(
            handleLocationUpdate,
            handleLocationError,
            {
              enableHighAccuracy: true,
              timeout: 10000,
              maximumAge: 0,
            }
          );
        }
      })
      .catch(() => {
        // Fallback if permissions API not available
        watchIdRef.current = navigator.geolocation.watchPosition(
          handleLocationUpdate,
          handleLocationError,
          {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 0,
          }
        );
      });
  };

  const stopTracking = () => {
    if (watchIdRef.current !== null) {
      navigator.geolocation.clearWatch(watchIdRef.current);
      watchIdRef.current = null;
    }
    if (updateIntervalRef.current !== null) {
      clearInterval(updateIntervalRef.current);
      updateIntervalRef.current = null;
    }
    setLocation(null);
  };

  useEffect(() => {
    if (isActive && deviceId) {
      startTracking();
    } else {
      stopTracking();
    }

    return () => {
      stopTracking();
    };
  }, [isActive, deviceId]);

  if (!isActive) {
    return null;
  }

  return (
    <div className="bg-gray-800 p-4 rounded-lg mb-4">
      <h3 className="text-lg font-semibold mb-2">GPS Location</h3>
      {permissionStatus === "denied" && (
        <p className="text-red-400">
          Location permission denied. Please enable in browser settings.
        </p>
      )}
      {error && !location && <p className="text-yellow-400">{error}</p>}
      {location ? (
        <div className="text-sm">
          <p className="text-green-400 flex items-center gap-2 mb-1">
            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
            GPS Active
          </p>
          <p className="text-gray-300">
            Lat: {location.lat.toFixed(6)}, Lng: {location.lng.toFixed(6)}
          </p>
        </div>
      ) : (
        <p className="text-gray-400 text-sm">Waiting for GPS signal...</p>
      )}
    </div>
  );
}
