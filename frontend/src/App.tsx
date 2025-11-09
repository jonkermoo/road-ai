import { useState } from "react";
import { SessionManager } from "./SessionManager";
import { LocationProvider } from "./LocationProvider";
import { EventMap } from "./EventMap";
import { BACKEND_URL } from "./config";

function App() {
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const [isSessionActive, setIsSessionActive] = useState(false);

  const handleSessionClaimed = (id: string) => {
    setDeviceId(id);
    setIsSessionActive(true);
  };

  const handleSessionLost = () => {
    setDeviceId(null);
    setIsSessionActive(false);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold mb-6 text-center">
          Road AI Dashboard
        </h1>

        {/* Session Management */}
        <SessionManager
          onSessionClaimed={handleSessionClaimed}
          onSessionLost={handleSessionLost}
        />

        {/* GPS Location Tracker */}
        <LocationProvider deviceId={deviceId} isActive={isSessionActive} />

        {/* Video Stream */}
        <div className="bg-gray-800 p-4 rounded-lg mb-4">
          <h2 className="text-2xl font-bold mb-4">Live Stream</h2>
          <div className="flex justify-center">
            <img
              src={`${BACKEND_URL}/video-feed`}
              alt="MJPEG Stream"
              className="border-4 border-gray-700 rounded-lg shadow-lg max-w-full h-auto"
              style={{ maxHeight: "480px" }}
            />
          </div>
        </div>

        {/* Event Map */}
        <EventMap />
      </div>
    </div>
  );
}

export default App;
