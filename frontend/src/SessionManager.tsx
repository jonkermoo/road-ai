import { useEffect, useState, useRef } from 'react';

interface SessionManagerProps {
  onSessionClaimed: (deviceId: string) => void;
  onSessionLost: () => void;
}

export function SessionManager({ onSessionClaimed, onSessionLost }: SessionManagerProps) {
  const [status, setStatus] = useState<'idle' | 'claiming' | 'active' | 'blocked'>('idle');
  const [deviceId] = useState(() => {
    // Generate or retrieve device ID from localStorage
    let id = localStorage.getItem('device_id');
    if (!id) {
      id = `device_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem('device_id', id);
    }
    return id;
  });

  const heartbeatIntervalRef = useRef<number | null>(null);

  const claimSession = async () => {
    setStatus('claiming');
    try {
      const res = await fetch('/session/claim', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ device_id: deviceId })
      });

      if (res.ok) {
        setStatus('active');
        onSessionClaimed(deviceId);
        startHeartbeat();
      } else if (res.status === 409) {
        setStatus('blocked');
        onSessionLost();
      }
    } catch (error) {
      console.error('Failed to claim session:', error);
      setStatus('idle');
    }
  };

  const sendHeartbeat = async () => {
    try {
      const res = await fetch('/session/heartbeat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ device_id: deviceId })
      });

      if (!res.ok) {
        console.error('Heartbeat failed');
        stopHeartbeat();
        setStatus('idle');
        onSessionLost();
      }
    } catch (error) {
      console.error('Heartbeat error:', error);
    }
  };

  const startHeartbeat = () => {
    if (heartbeatIntervalRef.current) return;
    heartbeatIntervalRef.current = window.setInterval(sendHeartbeat, 10000); // Every 10 seconds
  };

  const stopHeartbeat = () => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
  };

  const releaseSession = async () => {
    stopHeartbeat();
    try {
      await fetch('/session/release', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ device_id: deviceId })
      });
      setStatus('idle');
      onSessionLost();
    } catch (error) {
      console.error('Failed to release session:', error);
    }
  };

  useEffect(() => {
    // Try to claim session on mount
    claimSession();

    // Cleanup on unmount
    return () => {
      if (status === 'active') {
        releaseSession();
      }
    };
  }, []);

  return (
    <div className="bg-gray-800 p-4 rounded-lg mb-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Session Status</h3>
          <p className="text-sm text-gray-400">Device ID: {deviceId.substring(0, 20)}...</p>
        </div>
        <div className="flex items-center gap-2">
          {status === 'idle' && (
            <button
              onClick={claimSession}
              className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded"
            >
              Connect
            </button>
          )}
          {status === 'claiming' && (
            <span className="text-yellow-400">Connecting...</span>
          )}
          {status === 'active' && (
            <>
              <span className="text-green-400 flex items-center gap-2">
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                Active
              </span>
              <button
                onClick={releaseSession}
                className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded ml-2"
              >
                Disconnect
              </button>
            </>
          )}
          {status === 'blocked' && (
            <span className="text-red-400">Another device is streaming</span>
          )}
        </div>
      </div>
    </div>
  );
}
